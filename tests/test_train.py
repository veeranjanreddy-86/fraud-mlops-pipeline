from fraud_mlops.config import Settings
from fraud_mlops.train import time_split, train
from tests.conftest import FAST_PARAMS


def test_time_split_is_chronological(small_df):
    s = time_split(small_df)
    assert len(s.train) + len(s.valid) + len(s.test) == len(small_df)
    assert s.train["timestamp"].max() <= s.valid["timestamp"].min()
    assert s.valid["timestamp"].max() <= s.test["timestamp"].min()


def test_fast_training_beats_sanity_threshold(small_df):
    result = train(small_df, Settings(precision_target=0.8), params=FAST_PARAMS, log_to_mlflow=False)
    base_rate = small_df["is_fraud"].mean()
    assert result.test_metrics["pr_auc"] > 0.35 > 5 * base_rate
    assert result.test_metrics["roc_auc"] > 0.85
    assert 0.0 < result.threshold < 1.0


def test_smote_training_runs(small_df):
    result = train(small_df, Settings(), params=FAST_PARAMS, use_smote=True, log_to_mlflow=False)
    assert result.test_metrics["pr_auc"] > 0.3


def test_training_logs_and_registers(registered_champion, mlflow_settings):
    from mlflow.tracking import MlflowClient

    client = MlflowClient(mlflow_settings.tracking_uri, mlflow_settings.tracking_uri)
    run = client.get_run(registered_champion.run_id)
    for key in ("test_pr_auc", "test_roc_auc", "test_recall_at_precision", "decision_threshold", "test_tp"):
        assert key in run.data.metrics
    mv = client.get_model_version(mlflow_settings.model_name, registered_champion.model_version)
    assert float(mv.tags["decision_threshold"]) == round(registered_champion.threshold, 6)
