"""Central configuration, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"))
    artifact_root: str = field(
        default_factory=lambda: os.getenv("MLFLOW_ARTIFACT_ROOT", str(Path("mlruns/artifacts").resolve()))
    )
    experiment_name: str = field(default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT", "card-fraud"))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "fraud-detector"))
    champion_alias: str = field(default_factory=lambda: os.getenv("CHAMPION_ALIAS", "champion"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    precision_target: float = field(default_factory=lambda: _env_float("PRECISION_TARGET", 0.80))
    # Promotion-gate floors: a candidate must clear these regardless of the champion.
    min_pr_auc: float = field(default_factory=lambda: _env_float("GATE_MIN_PR_AUC", 0.40))
    min_recall_at_precision: float = field(default_factory=lambda: _env_float("GATE_MIN_RECALL_AT_PRECISION", 0.20))
    # Minimum improvement over the champion required to promote (absolute).
    min_improvement: float = field(default_factory=lambda: _env_float("GATE_MIN_IMPROVEMENT", 0.0))
    psi_retrain_threshold: float = field(default_factory=lambda: _env_float("PSI_RETRAIN_THRESHOLD", 0.2))


def get_settings() -> Settings:
    return Settings()
