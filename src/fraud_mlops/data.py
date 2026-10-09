"""Deterministic synthetic card-transaction generator.

The generator produces a realistic-looking but entirely synthetic dataset. Fraud labels are drawn
from a latent logistic model over the observable features plus unobserved noise, so the signal
is learnable but not trivially separable. The intercept is solved numerically so that the
expected fraud rate matches ``fraud_rate`` exactly, independent of the feature mix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES: dict[str, dict[str, float]] = {
    # share of traffic, median ticket (USD), log-odds risk uplift
    "grocery": {"share": 0.24, "median": 38.0, "risk": -0.6},
    "fuel": {"share": 0.12, "median": 45.0, "risk": -0.3},
    "restaurant": {"share": 0.18, "median": 32.0, "risk": -0.4},
    "retail": {"share": 0.16, "median": 70.0, "risk": 0.0},
    "travel": {"share": 0.06, "median": 320.0, "risk": 0.5},
    "electronics": {"share": 0.08, "median": 210.0, "risk": 0.9},
    "digital_goods": {"share": 0.09, "median": 25.0, "risk": 1.1},
    "gambling": {"share": 0.03, "median": 90.0, "risk": 1.3},
    "atm": {"share": 0.04, "median": 120.0, "risk": 0.7},
}

FEATURE_COLUMNS = [
    "amount",
    "merchant_category",
    "hour",
    "device_mismatch",
    "geo_mismatch",
    "txn_count_1h",
    "txn_count_24h",
    "account_age_days",
]
TARGET = "is_fraud"
TIMESTAMP = "timestamp"


def _solve_intercept(logit: np.ndarray, target_rate: float) -> float:
    """Find b such that mean(sigmoid(logit + b)) == target_rate (bisection, deterministic)."""
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(logit + mid)))))
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_transactions(
    n_rows: int = 50_000,
    fraud_rate: float = 0.015,
    seed: int = 42,
    start: str = "2025-01-01",
    days: int = 90,
    n_accounts: int = 8_000,
    drift: float = 0.0,
) -> pd.DataFrame:
    """Generate a synthetic transaction table sorted by timestamp.

    Args:
        n_rows: number of transactions.
        fraud_rate: expected share of fraudulent transactions.
        seed: RNG seed; identical arguments always yield an identical frame.
        start: first day of the observation window.
        days: length of the observation window.
        n_accounts: number of distinct synthetic card accounts.
        drift: 0..1 intensity of a population shift (bigger tickets, more digital goods,
            more device mismatches) used to simulate production drift.
    """
    if not 0.0 < fraud_rate < 0.5:
        raise ValueError("fraud_rate must be in (0, 0.5)")
    if not 0.0 <= drift <= 1.0:
        raise ValueError("drift must be in [0, 1]")

    rng = np.random.default_rng(seed)

    # Accounts and their ages.
    account_age = np.clip(rng.exponential(scale=900, size=n_accounts), 1, 7_000).astype(int)
    account_idx = rng.integers(0, n_accounts, size=n_rows)

    # Merchant mix (optionally shifted towards digital goods / electronics).
    cats = list(MERCHANT_CATEGORIES)
    shares = np.array([MERCHANT_CATEGORIES[c]["share"] for c in cats])
    if drift:
        boost = np.array([1.0 + 2.0 * drift if c in {"digital_goods", "electronics"} else 1.0 for c in cats])
        shares = shares * boost
    shares = shares / shares.sum()
    cat_idx = rng.choice(len(cats), size=n_rows, p=shares)
    category = np.array(cats)[cat_idx]

    medians = np.array([MERCHANT_CATEGORIES[c]["median"] for c in cats])[cat_idx]
    amount = np.round(medians * np.exp(rng.normal(0.0, 0.8, size=n_rows)) * (1.0 + 0.6 * drift), 2)
    amount = np.clip(amount, 0.5, 25_000.0)

    # Diurnal activity: mostly daytime with a thin night tail.
    hour_probs = np.array(
        [1, 0.6, 0.4, 0.3, 0.3, 0.5, 1.2, 2.5, 4, 5, 5.5, 6, 6.5, 6, 5.5, 5.5, 6, 6.5, 6.5, 6, 5, 4, 3, 2]
    )
    hour = rng.choice(24, size=n_rows, p=hour_probs / hour_probs.sum())

    device_mismatch = rng.random(n_rows) < (0.05 + 0.10 * drift)
    # Geo mismatch is correlated with device mismatch (new device while travelling / takeover).
    geo_mismatch = rng.random(n_rows) < np.where(device_mismatch, 0.25, 0.03)

    txn_count_1h = rng.poisson(0.6, size=n_rows)
    txn_count_24h = txn_count_1h + rng.poisson(3.0, size=n_rows)
    age = account_age[account_idx]

    # Latent fraud log-odds (intercept solved below).
    cat_risk = np.array([MERCHANT_CATEGORIES[c]["risk"] for c in cats])[cat_idx]
    night = ((hour <= 5) | (hour >= 23)).astype(float)
    logit = (
        1.1 * np.log1p(amount / 50.0)
        + 1.5 * cat_risk
        + 2.4 * device_mismatch
        + 1.9 * geo_mismatch
        + 1.2 * (device_mismatch & geo_mismatch)
        + 0.6 * txn_count_1h
        + 0.05 * (txn_count_24h - txn_count_1h)
        + 1.2 * night
        - 1.2 * np.log1p(age / 30.0)
        + rng.normal(0.0, 0.6, size=n_rows)  # unobserved factors
    )
    b0 = _solve_intercept(logit, fraud_rate)
    p_fraud = 1.0 / (1.0 + np.exp(-(logit + b0)))
    is_fraud = (rng.random(n_rows) < p_fraud).astype(int)

    # Fraudsters probe in bursts: inflate short-window velocity for some fraud rows.
    burst = (is_fraud == 1) & (rng.random(n_rows) < 0.45)
    extra = rng.poisson(3.0, size=n_rows) * burst
    txn_count_1h = txn_count_1h + extra
    txn_count_24h = txn_count_24h + extra

    seconds = np.sort(rng.integers(0, days * 86_400, size=n_rows))
    day_floor = (seconds // 86_400) * 86_400
    seconds = day_floor + hour * 3_600 + rng.integers(0, 3_600, size=n_rows)
    timestamp = pd.Timestamp(start) + pd.to_timedelta(seconds, unit="s")

    df = pd.DataFrame(
        {
            "transaction_id": [f"txn_{seed}_{i:07d}" for i in range(n_rows)],
            TIMESTAMP: timestamp,
            "account_id": [f"acct_{i:05d}" for i in account_idx],
            "amount": amount,
            "merchant_category": category,
            "hour": hour.astype(int),
            "device_mismatch": device_mismatch.astype(int),
            "geo_mismatch": geo_mismatch.astype(int),
            "txn_count_1h": txn_count_1h.astype(int),
            "txn_count_24h": txn_count_24h.astype(int),
            "account_age_days": age.astype(int),
            TARGET: is_fraud,
        }
    )
    return df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)


def save(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
