"""Feature pipeline: derived features + sklearn ColumnTransformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud_mlops.data import FEATURE_COLUMNS, MERCHANT_CATEGORIES

NUMERIC_FEATURES = [
    "log_amount",
    "hour_sin",
    "hour_cos",
    "txn_count_1h",
    "txn_count_24h",
    "velocity_ratio",
    "log_account_age",
]
BINARY_FEATURES = ["device_mismatch", "geo_mismatch", "is_night", "new_account"]
CATEGORICAL_FEATURES = ["merchant_category"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless, row-wise feature engineering (safe to run identically at train and serve time)."""
    out = df[FEATURE_COLUMNS].copy()
    out["log_amount"] = np.log1p(out["amount"].astype(float))
    hour = out["hour"].astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_night"] = ((hour <= 5) | (hour >= 23)).astype(int)
    out["velocity_ratio"] = out["txn_count_1h"] / (out["txn_count_24h"] + 1.0)
    out["log_account_age"] = np.log1p(out["account_age_days"].astype(float))
    out["new_account"] = (out["account_age_days"] < 30).astype(int)
    out["device_mismatch"] = out["device_mismatch"].astype(int)
    out["geo_mismatch"] = out["geo_mismatch"].astype(int)
    return out


def build_preprocessor() -> Pipeline:
    """Derived features followed by a ColumnTransformer producing a dense numeric matrix."""
    columns = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[sorted(MERCHANT_CATEGORIES)],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("derive", FunctionTransformer(add_derived_features, validate=False)),
            ("columns", columns),
        ]
    )


def n_output_features() -> int:
    return len(NUMERIC_FEATURES) + len(BINARY_FEATURES) + len(MERCHANT_CATEGORIES)
