import pandas as pd
import pytest

from fraud_mlops.data import FEATURE_COLUMNS, MERCHANT_CATEGORIES, TARGET, generate_transactions


@pytest.fixture(scope="module")
def full_df():
    return generate_transactions()  # default: 50k rows, 1.5% fraud, seed 42


def test_generator_is_deterministic():
    a = generate_transactions(n_rows=3_000, seed=123)
    b = generate_transactions(n_rows=3_000, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = generate_transactions(n_rows=3_000, seed=1)
    b = generate_transactions(n_rows=3_000, seed=2)
    assert not a["amount"].equals(b["amount"])


def test_default_size_and_fraud_rate(full_df):
    assert len(full_df) == 50_000
    assert 0.01 <= full_df[TARGET].mean() <= 0.02


def test_schema_and_ranges(full_df):
    assert {*FEATURE_COLUMNS, TARGET, "timestamp", "transaction_id"}.issubset(full_df.columns)
    assert full_df["timestamp"].is_monotonic_increasing
    assert full_df["transaction_id"].is_unique
    assert (full_df["amount"] > 0).all()
    assert full_df["hour"].between(0, 23).all()
    assert (full_df["txn_count_24h"] >= full_df["txn_count_1h"]).all()
    assert set(full_df["merchant_category"]).issubset(MERCHANT_CATEGORIES)
    assert full_df[["device_mismatch", "geo_mismatch", TARGET]].isin([0, 1]).all().all()


def test_fraud_is_associated_with_risk_signals(full_df):
    by_device = full_df.groupby("device_mismatch")[TARGET].mean()
    assert by_device[1] > 3 * by_device[0]


def test_drift_shifts_amounts():
    base = generate_transactions(n_rows=5_000, seed=5)
    drifted = generate_transactions(n_rows=5_000, seed=5, drift=1.0)
    assert drifted["amount"].median() > 1.3 * base["amount"].median()


@pytest.mark.parametrize("kwargs", [{"fraud_rate": 0.0}, {"fraud_rate": 0.7}, {"drift": 1.5}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_transactions(n_rows=100, **kwargs)
