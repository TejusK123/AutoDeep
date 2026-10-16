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
