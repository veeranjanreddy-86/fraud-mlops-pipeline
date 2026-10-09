"""Model training with time-based validation, threshold selection and MLflow tracking."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_mlops import __version__
from fraud_mlops.config import Settings
from fraud_mlops.data import FEATURE_COLUMNS, TARGET, TIMESTAMP
from fraud_mlops.features import build_preprocessor
from fraud_mlops.metrics import evaluate, threshold_for_cost, threshold_for_precision
from fraud_mlops.registry import THRESHOLD_TAG, setup_mlflow

log = logging.getLogger(__name__)

# MLflow 3 serialises sklearn models with skops, which refuses to load non-sklearn types unless
# they are explicitly allow-listed. Keep this list minimal and explicit.
SKOPS_TRUSTED_TYPES = [
    "fraud_mlops.features.add_derived_features",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]
SKOPS_TRUSTED_TYPES_SMOTE = [
    "imblearn.pipeline.Pipeline",
    "imblearn.over_sampling._smote.base.SMOTE",
]

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
}


@dataclass
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


@dataclass
class TrainResult:
    run_id: str | None
    model_version: str | None
    threshold: float
    valid_metrics: dict[str, float]
    test_metrics: dict[str, float]
    model: Any = field(repr=False)


def time_split(df: pd.DataFrame, valid_frac: float = 0.15, test_frac: float = 0.15) -> Splits:
    """Chronological split: oldest -> train, middle -> threshold tuning, newest -> holdout test.

    A random split would leak future behaviour (e.g. later velocity patterns) into training and
    overstate performance compared to a model that is always scoring tomorrow's traffic.
    """
    df = df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)
    n = len(df)
    i_valid = int(n * (1 - valid_frac - test_frac))
    i_test = int(n * (1 - test_frac))
    return Splits(df.iloc[:i_valid], df.iloc[i_valid:i_test], df.iloc[i_test:])


def build_model(params: dict[str, Any], scale_pos_weight: float, use_smote: bool, seed: int):
    clf = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        # imblearn forbids nested Pipelines, so the preprocessing steps are inlined. SMOTE runs
        # only during fit; at predict time the sampler is a no-op.
        return ImbPipeline(
            [
                *build_preprocessor().steps,
                ("smote", SMOTE(sampling_strategy=0.1, random_state=seed)),
                ("model", clf),
            ]
        )
    return Pipeline([("features", build_preprocessor()), ("model", clf)])


def train(
    df: pd.DataFrame,
    settings: Settings,
    params: dict[str, Any] | None = None,
    threshold_method: str = "precision",
    use_smote: bool = False,
    seed: int = 42,
    register: bool = True,
    log_to_mlflow: bool = True,
    run_name: str | None = None,
) -> TrainResult:
    params = {**DEFAULT_PARAMS, **(params or {})}
    splits = time_split(df)
    x_tr, y_tr = splits.train[FEATURE_COLUMNS], splits.train[TARGET].to_numpy()
    x_va, y_va = splits.valid[FEATURE_COLUMNS], splits.valid[TARGET].to_numpy()
    x_te, y_te = splits.test[FEATURE_COLUMNS], splits.test[TARGET].to_numpy()

    # Class-imbalance handling: weight positives by the neg/pos ratio. With SMOTE the minority is
    # already oversampled to 10% of the majority, so the weight is reduced accordingly.
    pos = max(int(y_tr.sum()), 1)
    spw = (len(y_tr) - pos) / pos
    if use_smote:
        spw = max(spw * pos / (0.1 * (len(y_tr) - pos)), 1.0)

    model = build_model(params, spw, use_smote, seed)
    model.fit(x_tr, y_tr)

    p_va = model.predict_proba(x_va)[:, 1]
    if threshold_method == "precision":
        threshold = threshold_for_precision(y_va, p_va, settings.precision_target)
    elif threshold_method == "cost":
        threshold = threshold_for_cost(y_va, p_va)
    else:
        raise ValueError(f"unknown threshold_method {threshold_method!r}")

    valid_metrics = evaluate(y_va, p_va, threshold, settings.precision_target)
    p_te = model.predict_proba(x_te)[:, 1]
    test_metrics = evaluate(y_te, p_te, threshold, settings.precision_target)
    log.info("valid=%s test=%s threshold=%.4f", valid_metrics, test_metrics, threshold)

    result = TrainResult(None, None, threshold, valid_metrics, test_metrics, model)
    if not log_to_mlflow:
        return result

    client = setup_mlflow(settings)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"package_version": __version__, "model_family": "xgboost"})
        mlflow.log_params(
            {
                **params,
                "scale_pos_weight": round(spw, 3),
                "use_smote": use_smote,
                "threshold_method": threshold_method,
                "precision_target": settings.precision_target,
                "seed": seed,
                "n_train": len(y_tr),
                "n_valid": len(y_va),
                "n_test": len(y_te),
                "train_fraud_rate": round(float(y_tr.mean()), 5),
                "train_end": str(splits.train[TIMESTAMP].max()),
                "test_start": str(splits.test[TIMESTAMP].min()),
            }
        )
        mlflow.log_metric("decision_threshold", threshold)
        mlflow.log_metrics({f"valid_{k}": v for k, v in valid_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        with tempfile.TemporaryDirectory() as tmp:
            cm = {
                split: {"labels": ["legit", "fraud"], "matrix": [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]}
                for split, m in (("valid", valid_metrics), ("test", test_metrics))
            }
            cm_path = Path(tmp) / "confusion_matrix.json"
            cm_path.write_text(json.dumps(cm, indent=2))
            mlflow.log_artifact(str(cm_path))

        sample = x_va.head(20)
        signature = infer_signature(sample, model.predict_proba(sample)[:, 1])
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=sample.head(3),
            pyfunc_predict_fn="predict_proba",
            pip_requirements=_pip_requirements(use_smote),
            skops_trusted_types=SKOPS_TRUSTED_TYPES + (SKOPS_TRUSTED_TYPES_SMOTE if use_smote else []),
            registered_model_name=settings.model_name if register else None,
        )
        result.run_id = run.info.run_id

    if register:
        version = str(info.registered_model_version)
        client.set_model_version_tag(settings.model_name, version, THRESHOLD_TAG, f"{threshold:.6f}")
        client.set_model_version_tag(settings.model_name, version, "precision_target", str(settings.precision_target))
        client.set_model_version_tag(settings.model_name, version, "gate_status", "pending")
        result.model_version = version
        log.info("registered %s version %s", settings.model_name, version)
    return result


def _pip_requirements(use_smote: bool) -> list[str]:
    import sklearn
    import xgboost

    reqs = [f"scikit-learn=={sklearn.__version__}", f"xgboost=={xgboost.__version__}", "pandas", "numpy"]
    if use_smote:
        import imblearn

        reqs.append(f"imbalanced-learn=={imblearn.__version__}")
    return reqs


def summarize(result: TrainResult) -> dict[str, Any]:
    keys = ("roc_auc", "pr_auc", "recall_at_precision", "precision", "recall", "alert_rate")
    return {
        "run_id": result.run_id,
        "model_version": result.model_version,
        "threshold": round(result.threshold, 4),
        "test": {k: round(result.test_metrics[k], 4) for k in keys}
        | {k: result.test_metrics[k] for k in ("tp", "fp", "fn", "tn")},
    }


__all__ = ["TrainResult", "summarize", "time_split", "train"]
