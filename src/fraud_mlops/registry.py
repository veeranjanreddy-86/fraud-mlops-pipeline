"""Thin helpers around the MLflow tracking server and model registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from fraud_mlops.config import Settings

THRESHOLD_TAG = "decision_threshold"


def setup_mlflow(settings: Settings) -> MlflowClient:
    """Point MLflow at the configured (local by default) tracking store and ensure the experiment."""
    uri = settings.tracking_uri
    if uri.startswith("sqlite:///"):
        db_path = Path(uri.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    if client.get_experiment_by_name(settings.experiment_name) is None:
        Path(settings.artifact_root).mkdir(parents=True, exist_ok=True)
        client.create_experiment(
            settings.experiment_name, artifact_location=Path(settings.artifact_root).resolve().as_uri()
        )
    mlflow.set_experiment(settings.experiment_name)
    return client


@dataclass
class ModelBundle:
    """A fitted pipeline plus the metadata needed to make and explain decisions."""

    model: Any
    threshold: float
    name: str
    version: str
    source: str

    def predict_proba(self, frame) -> Any:
        return self.model.predict_proba(frame)[:, 1]


def latest_version(client: MlflowClient, name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise LookupError(f"No versions registered for model '{name}'")
    return str(max(int(v.version) for v in versions))


def champion_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except Exception:  # noqa: BLE001 - MLflow raises a generic RestException when alias is unset
        return None


def load_version(client: MlflowClient, name: str, version: str) -> ModelBundle:
    mv = client.get_model_version(name, version)
    model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    threshold = float(mv.tags.get(THRESHOLD_TAG, 0.5))
    return ModelBundle(model, threshold, name, str(version), source=f"models:/{name}/{version}")


def load_alias(client: MlflowClient, name: str, alias: str) -> ModelBundle:
    mv = client.get_model_version_by_alias(name, alias)
    bundle = load_version(client, name, str(mv.version))
    bundle.source = f"models:/{name}@{alias}"
    return bundle


def export_bundle(bundle: ModelBundle, out_dir: str | Path) -> Path:
    """Write a registry-independent copy (joblib + metadata) for container images."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.model, out / "model.joblib")
    meta = {"name": bundle.name, "version": bundle.version, "threshold": bundle.threshold}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def load_exported(path: str | Path) -> ModelBundle:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    model = joblib.load(path / "model.joblib")
    return ModelBundle(model, float(meta["threshold"]), meta["name"], str(meta["version"]), source=str(path))
