import pytest

from fraud_mlops.config import Settings
from fraud_mlops.gate import decide

SETTINGS = Settings(min_pr_auc=0.4, min_recall_at_precision=0.2, min_improvement=0.0)
CHAMP = {"pr_auc": 0.55, "recall_at_precision": 0.35}


def test_promotes_first_model_meeting_floors():
    ok, reasons = decide({"pr_auc": 0.5, "recall_at_precision": 0.3}, None, SETTINGS)
    assert ok and "no current champion" in reasons[0]


@pytest.mark.parametrize(
    "candidate, needle",
    [
        ({"pr_auc": 0.35, "recall_at_precision": 0.3}, "pr_auc=0.3500 below minimum"),
        ({"pr_auc": 0.60, "recall_at_precision": 0.1}, "recall_at_precision=0.1000 below minimum"),
    ],
)
def test_rejects_below_floor(candidate, needle):
    ok, reasons = decide(candidate, None, SETTINGS)
    assert not ok and any(needle in r for r in reasons)


def test_promotes_strictly_better_candidate():
    ok, reasons = decide({"pr_auc": 0.60, "recall_at_precision": 0.40}, CHAMP, SETTINGS)
    assert ok and "improved" in reasons[0]


def test_rejects_regression_on_either_metric():
    ok, reasons = decide({"pr_auc": 0.60, "recall_at_precision": 0.30}, CHAMP, SETTINGS)
    assert not ok and "recall_at_precision regressed" in reasons[0]


def test_rejects_insufficient_improvement():
    strict = Settings(min_pr_auc=0.4, min_recall_at_precision=0.2, min_improvement=0.02)
    ok, reasons = decide({"pr_auc": 0.56, "recall_at_precision": 0.36}, CHAMP, strict)
    assert not ok and "does not exceed" in reasons[0]


def test_gate_end_to_end_rejects_identical_retrain(registered_champion, small_df, mlflow_settings):
    """Retraining with identical data/seed produces an equal model, which must not displace the champion."""
    from mlflow.tracking import MlflowClient

    from fraud_mlops.gate import run_gate
    from fraud_mlops.train import train
    from tests.conftest import FAST_PARAMS

    client = MlflowClient(mlflow_settings.tracking_uri, mlflow_settings.tracking_uri)
    champ_before = client.get_model_version_by_alias(mlflow_settings.model_name, "champion").version

    candidate = train(small_df, mlflow_settings, params=FAST_PARAMS, run_name="test-v2")
    decision = run_gate(small_df, mlflow_settings, candidate_version=candidate.model_version)

    assert not decision.promote
    assert decision.champion_version == str(champ_before)
    mv = client.get_model_version(mlflow_settings.model_name, candidate.model_version)
    assert mv.tags["gate_status"] == "rejected"
    assert client.get_model_version_by_alias(mlflow_settings.model_name, "champion").version == champ_before
