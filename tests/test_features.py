import numpy as np

from fraud_mlops.data import FEATURE_COLUMNS
from fraud_mlops.features import add_derived_features, build_preprocessor, n_output_features


def test_preprocessor_output_shape(small_df):
    pre = build_preprocessor()
    x = pre.fit_transform(small_df[FEATURE_COLUMNS])
    assert x.shape == (len(small_df), n_output_features())
    assert np.isfinite(x).all()


def test_feature_names_are_stable(small_df):
    pre = build_preprocessor().fit(small_df[FEATURE_COLUMNS])
    names = list(pre.named_steps["columns"].get_feature_names_out())
    assert len(names) == n_output_features()
    assert "merchant_category_gambling" in names


def test_unknown_category_is_ignored(small_df):
    pre = build_preprocessor().fit(small_df[FEATURE_COLUMNS])
    row = small_df[FEATURE_COLUMNS].head(1).copy()
    row["merchant_category"] = "crypto_exchange"
    x = pre.transform(row)
    assert x.shape == (1, n_output_features())


def test_derived_features_are_row_wise(small_df):
    head = small_df.head(50)
    full = add_derived_features(small_df).head(50)
    part = add_derived_features(head)
    np.testing.assert_allclose(full["log_amount"].to_numpy(), part["log_amount"].to_numpy())
