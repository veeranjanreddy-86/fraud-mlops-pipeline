import os
from pathlib import Path

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from fraud_mlops.config import Settings
from fraud_mlops.data import generate_transactions

FAST_PARAMS = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1}


@pytest.fixture(scope="session")
def small_df():
    # Higher fraud rate on a small sample keeps tests fast while leaving enough positives to evaluate.
    return generate_transactions(n_rows=12_000, fraud_rate=0.03, seed=7)


@pytest.fixture(scope="session")
def mlflow_settings(tmp_path_factory) -> Settings:
    root: Path = tmp_path_factory.mktemp("mlflow")
    return Settings(
        tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
        artifact_root=str(root / "artifacts"),
        experiment_name="test-fraud",
        model_name="fraud-detector-test",
        champion_alias="champion",
        data_dir=root,
        precision_target=0.8,
        min_pr_auc=0.3,
        min_recall_at_precision=0.1,
        min_improvement=0.0,
    )


@pytest.fixture(scope="session")
def registered_champion(small_df, mlflow_settings):
    """Train + register v1 in a throwaway MLflow store and promote it via the gate."""
    from fraud_mlops.gate import run_gate
    from fraud_mlops.train import train

    result = train(small_df, mlflow_settings, params=FAST_PARAMS, run_name="test-v1")
    decision = run_gate(small_df, mlflow_settings, candidate_version=result.model_version)
    assert decision.promote, decision.reasons
    return result
