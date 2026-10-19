"""SHAP values of the model's predictions, from XGBoost's built-in TreeSHAP (C4).

A feature's SHAP value for a locus and class is how much that feature moves the
model's raw score for the class (its log-odds before the softmax) away from the
average. The values are exact, and with the bias they add up to the raw score.
"""
import numpy as np
import pandas as pd
import xgboost

TOP = 3


def contributions(booster, X):
    """Return SHAP values, shape (loci, classes, features), and the bias, shape (classes,).

    X holds the model's features (booster.feature_names, in that order); blank
    values are allowed, as in prediction.
    """
    values = booster.predict(xgboost.DMatrix(X, feature_names=booster.feature_names), pred_contribs=True)
    return values[:, :, :-1], values[0, :, -1]


def top_contributions(values, predicted, names, k=TOP):
    """For each locus, the k features with the largest SHAP values by size for its predicted class.

    Returns columns shap_top1_feature, shap_top1_value, ... The values keep
    their sign: positive pushes towards the predicted class, negative away.
    """
    own = values[np.arange(len(values)), predicted]
    order = np.argsort(-np.abs(own), axis=1, kind="stable")[:, :k]
    columns = {}
    for rank in range(k):
        columns[f"shap_top{rank + 1}_feature"] = [names[j] for j in order[:, rank]]
        columns[f"shap_top{rank + 1}_value"] = own[np.arange(len(own)), order[:, rank]]
    return pd.DataFrame(columns)
