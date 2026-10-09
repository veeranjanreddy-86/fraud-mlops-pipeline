import pytest
from fastapi.testclient import TestClient

from fraud_mlops.registry import export_bundle, load_alias, load_exported, setup_mlflow
from fraud_mlops.serve import create_app

TXN = {
    "transaction_id": "t-1",
    "amount": 129.99,
    "merchant_category": "electronics",
    "hour": 3,
    "device_mismatch": True,
    "geo_mismatch": True,
    "txn_count_1h": 6,
    "txn_count_24h": 9,
    "account_age_days": 4,
}
SAFE = TXN | {
    "transaction_id": "t-2",
    "merchant_category": "grocery",
    "hour": 11,
    "amount": 23.5,
    "device_mismatch": False,
    "geo_mismatch": False,
    "txn_count_1h": 0,
    "txn_count_24h": 2,
    "account_age_days": 2000,
}


@pytest.fixture(scope="module")
def bundle(registered_champion, mlflow_settings):
    client = setup_mlflow(mlflow_settings)
    return load_alias(client, mlflow_settings.model_name, mlflow_settings.champion_alias)


@pytest.fixture(scope="module")
def client(bundle):
    with TestClient(create_app(bundle)) as c:
        yield c


def test_health(client, bundle):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == bundle.version
    assert body["model_source"].endswith("@champion")


def test_score_single(client, bundle):
    r = client.post("/score", json=TXN)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"approve", "decline"}
    assert body["threshold"] == pytest.approx(bundle.threshold, abs=1e-6)
    assert body["model_version"] == bundle.version
    assert "x-latency-ms" in r.headers and "x-request-id" in r.headers


def test_risky_transaction_scores_higher_than_safe(client):
    risky = client.post("/score", json=TXN).json()["fraud_probability"]
    safe = client.post("/score", json=SAFE).json()["fraud_probability"]
    assert risky > safe


def test_score_batch(client):
    r = client.post("/score/batch", json={"transactions": [TXN, SAFE]})
    assert r.status_code == 200
    body = r.json()
    assert [s["transaction_id"] for s in body["scores"]] == ["t-1", "t-2"]
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize(
    "patch",
    [
        {"amount": -5},
        {"merchant_category": "crypto"},
        {"hour": 24},
        {"txn_count_1h": 5, "txn_count_24h": 1},
        {"unexpected": "field"},
    ],
)
def test_validation_errors(client, patch):
    assert client.post("/score", json=TXN | patch).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/score/batch", json={"transactions": []}).status_code == 422


def test_exported_bundle_serves_identically(bundle, tmp_path, monkeypatch):
    export_bundle(bundle, tmp_path / "champion")
    reloaded = load_exported(tmp_path / "champion")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "champion"))
    with TestClient(create_app()) as c:  # loads from MODEL_PATH at startup
        body = c.post("/score", json=TXN).json()
    assert body["model_version"] == bundle.version
    assert body["fraud_probability"] == pytest.approx(
        float(reloaded.predict_proba(__import__("pandas").DataFrame([TXN]).drop(columns="transaction_id"))[0]),
        abs=1e-6,
    )
