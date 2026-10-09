"""Champion/challenger promotion gate driven by MLflow model aliases.

The candidate and the current champion are scored on the *same* most-recent holdout window so
the comparison is apples-to-apples, then the decision rules in :func:`decide` are applied.
Promotion is a metadata operation (moving the ``champion`` alias), which makes rollback a
one-liner: point the alias back at the version tagged ``previous_champion``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fraud_mlops.config import Settings
from fraud_mlops.data import FEATURE_COLUMNS, TARGET
from fraud_mlops.metrics import recall_at_precision
from fraud_mlops.registry import champion_version, latest_version, load_version, setup_mlflow
from fraud_mlops.train import time_split

log = logging.getLogger(__name__)

GATED_METRICS = ("pr_auc", "recall_at_precision")
PREVIOUS_ALIAS = "previous_champion"


@dataclass
class GateDecision:
    promote: bool
    candidate_version: str | None
    champion_version: str | None
    candidate_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None
    reasons: list[str] = field(default_factory=list)
    decided_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


def decide(
    candidate: dict[str, float],
    champion: dict[str, float] | None,
    settings: Settings,
) -> tuple[bool, list[str]]:
    """Pure decision function (no I/O) so the rules are unit-testable.

    Rules:
      1. Candidate must meet absolute floors for PR-AUC and recall@precision.
      2. If a champion exists, the candidate must not regress on either gated metric and must
         beat the champion on PR-AUC by at least ``settings.min_improvement``.
    """
    reasons: list[str] = []
    floors = {"pr_auc": settings.min_pr_auc, "recall_at_precision": settings.min_recall_at_precision}
    for metric, floor in floors.items():
        if candidate[metric] < floor:
            reasons.append(f"{metric}={candidate[metric]:.4f} below minimum {floor:.4f}")
    if reasons:
        return False, reasons

    if champion is None:
        return True, ["no current champion; candidate meets minimum thresholds"]

    for metric in GATED_METRICS:
        if candidate[metric] < champion[metric]:
            reasons.append(f"{metric} regressed: candidate {candidate[metric]:.4f} < champion {champion[metric]:.4f}")
    gain = candidate["pr_auc"] - champion["pr_auc"]
    if not reasons and gain <= settings.min_improvement:
        reasons.append(f"pr_auc gain {gain:+.4f} does not exceed required improvement {settings.min_improvement:+.4f}")
    if reasons:
        return False, reasons
    return True, [
        f"pr_auc improved {champion['pr_auc']:.4f} -> {candidate['pr_auc']:.4f}; "
        f"recall@precision {champion['recall_at_precision']:.4f} -> {candidate['recall_at_precision']:.4f}"
    ]


def _holdout_metrics(bundle, holdout: pd.DataFrame, target_precision: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score

    y = holdout[TARGET].to_numpy()
    p = bundle.predict_proba(holdout[FEATURE_COLUMNS])
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "recall_at_precision": recall_at_precision(y, p, target_precision),
    }


def run_gate(
    df: pd.DataFrame,
    settings: Settings,
    candidate_version: str | None = None,
    report_path: str | Path | None = None,
) -> GateDecision:
    client = setup_mlflow(settings)
    name = settings.model_name
    candidate_version = candidate_version or latest_version(client, name)
    current = champion_version(client, name, settings.champion_alias)
    holdout = time_split(df).test

    cand_metrics = _holdout_metrics(load_version(client, name, candidate_version), holdout, settings.precision_target)

    if current == candidate_version:
        decision = GateDecision(
            False, candidate_version, current, cand_metrics, cand_metrics, ["candidate is already the champion"]
        )
    else:
        champ_metrics = (
            _holdout_metrics(load_version(client, name, current), holdout, settings.precision_target)
            if current
            else None
        )
        promote, reasons = decide(cand_metrics, champ_metrics, settings)
        decision = GateDecision(promote, candidate_version, current, cand_metrics, champ_metrics, reasons)

        status = "promoted" if promote else "rejected"
        client.set_model_version_tag(name, candidate_version, "gate_status", status)
        client.set_model_version_tag(name, candidate_version, "gate_reason", "; ".join(reasons)[:500])
        for metric, value in cand_metrics.items():
            client.set_model_version_tag(name, candidate_version, f"gate_{metric}", f"{value:.6f}")
        if promote:
            if current:
                client.set_registered_model_alias(name, PREVIOUS_ALIAS, current)
            client.set_registered_model_alias(name, settings.champion_alias, candidate_version)
            log.info("promoted %s v%s to @%s", name, candidate_version, settings.champion_alias)
        else:
            log.info("rejected %s v%s: %s", name, candidate_version, reasons)

    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decision.to_dict(), indent=2))
    return decision
