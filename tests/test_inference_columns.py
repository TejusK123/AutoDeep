"""Inference must match input columns to the model's features by name."""
import os
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")
INFERENCE = os.path.join(PKG, "scripts", "boosted_forest_inference.py")
TRAINING_CSV = os.path.join(PKG, "training_data_pre", "original_data_Pomonella.csv")


def run_inference(features, workdir):
    run_dir = os.path.join(workdir, "AutoDeepRun")
    os.makedirs(run_dir, exist_ok=True)
    features.to_csv(os.path.join(run_dir, "fully_formatted_data.csv"), index=False)
    result = subprocess.run([sys.executable, INFERENCE], cwd=workdir, capture_output=True, text=True)
    out = os.path.join(run_dir, "xgboost_judgement.csv")
    return result, (pd.read_csv(out) if result.returncode == 0 else None)


def features():
    return pd.read_csv(TRAINING_CSV).iloc[:, :-1]


def test_column_order_does_not_matter(tmp_path):
    df = features()
    _, ordered = run_inference(df, str(tmp_path / "a"))
    shuffled = df[[df.columns[0]] + list(df.columns[1:][::-1])]
    _, reordered = run_inference(shuffled, str(tmp_path / "b"))
    assert list(reordered["prediction"]) == list(ordered["prediction"])
    assert (reordered["Confident_Prob"] - ordered["Confident_Prob"]).abs().max() < 1e-6


def test_missing_model_feature_is_an_error(tmp_path):
    df = features().rename(columns={"num_locis": "normalized_entropy_s"})
    result, out = run_inference(df, str(tmp_path))
    assert result.returncode != 0
    assert "num_locis" in result.stderr


def test_output_lists_each_locus_top_three_contributions(tmp_path):
    # C4: the three features with the largest SHAP values (by size) for the predicted class
    import pytest
    import xgboost

    from explain import contributions
    from features import LABELS, encode_features

    df = features()
    result, out = run_inference(df, str(tmp_path))
    assert result.returncode == 0, result.stderr
    booster = xgboost.Booster()
    booster.load_model(os.path.join(PKG, "model_weights", "miRNA_model.json"))
    names = booster.feature_names
    values, _ = contributions(booster, encode_features(df)[names])
    for i, row in out.iterrows():
        own = values[i, LABELS.index(row["prediction"])]
        expected = sorted(range(len(names)), key=lambda j: -abs(own[j]))[:3]
        assert [row[f"shap_top{k}_feature"] for k in (1, 2, 3)] == [names[j] for j in expected]
        assert [row[f"shap_top{k}_value"] for k in (1, 2, 3)] == pytest.approx([own[j] for j in expected], abs=1e-6)
