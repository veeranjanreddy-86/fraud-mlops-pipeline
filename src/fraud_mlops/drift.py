"""Feature drift monitoring with Population Stability Index (PSI) and Kolmogorov-Smirnov tests.

PSI rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift (retrain).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from fraud_mlops.data import FEATURE_COLUMNS

EPS = 1e-6
CATEGORICAL = {"merchant_category", "device_mismatch", "geo_mismatch"}


def psi_from_proportions(expected: np.ndarray, actual: np.ndarray, eps: float = EPS) -> float:
    """PSI = sum((a - e) * ln(a / e)) over bins; empty bins are floored at ``eps``."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI using quantile bins learnt on the reference sample (robust to skewed features)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:  # constant reference: fall back to categorical treatment
        return psi_categorical(reference, current)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, edges)[0]
    cur_counts = np.histogram(current, edges)[0]
    return psi_from_proportions(ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum())


def psi_categorical(reference, current) -> float:
    ref = pd.Series(reference).value_counts(normalize=True)
    cur = pd.Series(current).value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    return psi_from_proportions(ref.reindex(cats, fill_value=0).values, cur.reindex(cats, fill_value=0).values)


def status_for(psi: float, threshold: float = 0.2) -> str:
    if psi > threshold:
        return "significant"
    if psi > 0.1:
        return "moderate"
    return "stable"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    psi_threshold: float = 0.2,
    bins: int = 10,
) -> dict[str, Any]:
    features = features or FEATURE_COLUMNS
    per_feature: dict[str, dict[str, Any]] = {}
    for col in features:
        if col in CATEGORICAL:
            psi = psi_categorical(reference[col], current[col])
            ks_stat, ks_p = None, None
        else:
            psi = psi_numeric(reference[col].to_numpy(), current[col].to_numpy(), bins=bins)
            ks = ks_2samp(reference[col].to_numpy(), current[col].to_numpy(), method="asymp")
            ks_stat, ks_p = round(float(ks.statistic), 6), float(ks.pvalue)
        per_feature[col] = {
            "type": "categorical" if col in CATEGORICAL else "numeric",
            "psi": round(psi, 6),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "status": status_for(psi, psi_threshold),
        }
    drifted = [c for c, r in per_feature.items() if r["psi"] > psi_threshold]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_reference": len(reference),
        "n_current": len(current),
        "psi_threshold": psi_threshold,
        "retrain_recommended": bool(drifted),
        "drifted_features": drifted,
        "features": per_feature,
    }


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
