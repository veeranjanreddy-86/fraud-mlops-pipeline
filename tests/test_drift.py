import math

import numpy as np
import pytest

from fraud_mlops.data import generate_transactions
from fraud_mlops.drift import drift_report, psi_categorical, psi_from_proportions, psi_numeric


def test_psi_known_value():
    expected = [0.25, 0.25, 0.25, 0.25]
    actual = [0.10, 0.20, 0.30, 0.40]
    manual = sum((a - e) * math.log(a / e) for a, e in zip(actual, expected, strict=True))
    assert psi_from_proportions(expected, actual) == pytest.approx(manual, rel=1e-9)
    assert manual == pytest.approx(0.2282, abs=1e-4)


def test_psi_identical_distribution_is_zero():
    assert psi_from_proportions([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0, abs=1e-12)


def test_psi_numeric_same_vs_shifted_normal():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 50_000)
    same = rng.normal(0, 1, 50_000)
    shifted = rng.normal(1, 1, 50_000)
    assert psi_numeric(ref, same) < 0.01
    # A one-sigma mean shift is a large, well-known drift (theoretical PSI ~ 1.0 for continuous bins).
    assert psi_numeric(ref, shifted) > 0.2


def test_psi_categorical_symmetric_zero_and_new_category():
    assert psi_categorical(list("aabb"), list("abab")) == pytest.approx(0.0, abs=1e-12)
    assert psi_categorical(list("aaaabbbb"), list("aaaabbbc")) > 0.1  # unseen category is penalised


def test_drift_report_flags_shifted_batch():
    ref = generate_transactions(n_rows=10_000, seed=1)
    same = generate_transactions(n_rows=10_000, seed=2, start="2025-04-01")
    shifted = generate_transactions(n_rows=10_000, seed=3, start="2025-04-01", drift=1.0)

    stable = drift_report(ref, same)
    assert not stable["retrain_recommended"]
    assert all(f["status"] == "stable" for f in stable["features"].values())

    drifted = drift_report(ref, shifted)
    assert drifted["retrain_recommended"]
    assert "amount" in drifted["drifted_features"]
    assert drifted["features"]["amount"]["ks_pvalue"] < 1e-6
    assert drifted["features"]["merchant_category"]["ks_statistic"] is None
