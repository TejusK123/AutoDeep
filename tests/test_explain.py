"""C4: SHAP values from XGBoost's built-in TreeSHAP, and each locus's top contributions."""
import os

import numpy as np
import pandas as pd
import pytest
import xgboost

from features import encode_features

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")


@pytest.fixture(scope="module")
def shipped():
    booster = xgboost.Booster()
    booster.load_model(os.path.join(PKG, "model_weights", "miRNA_model.json"))
    data = pd.read_csv(os.path.join(PKG, "training_data_pre", "original_data_Pomonella.csv"))
    return booster, encode_features(data)[booster.feature_names]


def test_contributions_add_up_to_the_raw_scores(shipped):
    from explain import contributions

    booster, X = shipped
    values, bias = contributions(booster, X)
    assert values.shape == (len(X), 3, len(booster.feature_names))
    margin = booster.predict(xgboost.DMatrix(X, feature_names=booster.feature_names), output_margin=True)
    # TreeSHAP is exact: per class, the bias plus the contributions is the raw score (float32 sums)
    assert np.abs(values.sum(axis=2) + bias - margin).max() < 1e-4


def test_blank_features_get_contributions(shipped):
    from explain import contributions

    booster, X = shipped
    X = X.copy()
    X[["3' Overhang Bot", "3' Overhang Top"]] = np.nan
    values, _ = contributions(booster, X)
    assert np.isfinite(values).all()


def test_top_contributions_are_the_largest_for_the_predicted_class():
    from explain import top_contributions

    names = ["a", "b", "c", "d"]
    values = np.zeros((2, 3, 4))
    values[0, 1] = [0.1, -0.9, 0.5, 0.2]  # locus 0, predicted class 1
    values[0, 0] = [5, 5, 5, 5]           # other classes do not count
    values[1, 2] = [0.3, 0.0, -0.1, -0.4]  # locus 1, predicted class 2
    top = top_contributions(values, np.array([1, 2]), names)
    assert list(top.columns) == ["shap_top1_feature", "shap_top1_value", "shap_top2_feature", "shap_top2_value",
                                 "shap_top3_feature", "shap_top3_value"]
    assert list(top.iloc[0]) == ["b", -0.9, "c", 0.5, "d", 0.2]
    assert list(top.iloc[1]) == ["d", -0.4, "a", 0.3, "c", -0.1]


def test_every_feature_has_a_table1_name():
    from features import TABLE1_NAMES

    booster = xgboost.Booster()
    booster.load_model(os.path.join(PKG, "model_weights", "miRNA_model.json"))
    assert set(booster.feature_names) <= set(TABLE1_NAMES)
    assert len(set(TABLE1_NAMES.values())) == len(TABLE1_NAMES)
