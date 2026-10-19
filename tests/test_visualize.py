"""A5: tree plots are opt-in, dumped once, and labelled by round and class.

C4: SHAP plots are the default output, as vector files with the text kept as
text; the gain, weight and cover plots are a legacy option.
"""
import glob
import os
import re
import subprocess
import sys

import pandas as pd
import pytest
import xgboost

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")
VISUALIZE = os.path.join(PKG, "scripts", "boosted_forest_visualization.py")
TRAINING_CSV = os.path.join(PKG, "training_data_pre", "original_data_Pomonella.csv")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
           PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])


@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    from features import encode_features, encode_labels

    data = pd.read_csv(TRAINING_CSV)
    model = xgboost.XGBClassifier(n_estimators=3, max_depth=2)
    model.fit(encode_features(data.iloc[:, 1:-1]), encode_labels(data.iloc[:, -1]))
    path = str(tmp_path_factory.mktemp("model") / "small_model.json")
    model.save_model(path)
    return path


SHAP_FILES = ["shap_dependence_five_prime_homogeneity", "shap_dependence_folding_energy",
              "shap_dependence_overhang_bot", "shap_dependence_overhang_top", "shap_importance",
              "shap_summary_Candidate", "shap_summary_Confident", "shap_summary_falsepositive"]
LEGACY_FILES = ["feature_importance_cover.png", "feature_importance_gain.png", "feature_importance_weight.png"]


def visualize(cwd, *args):
    result = subprocess.run([sys.executable, VISUALIZE, *args], cwd=cwd, env=ENV, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    (out,) = glob.glob(os.path.join(cwd, "plots*"))
    return sorted(os.listdir(out))


def output_dir(cwd):
    (out,) = glob.glob(os.path.join(cwd, "plots*"))
    return out


def test_default_writes_shap_plots_and_tables(small_model, tmp_path):
    files = visualize(tmp_path, "--model", small_model)
    assert files == sorted([f + ".pdf" for f in SHAP_FILES] + ["shap_importance.csv", "shap_values.csv"])
    # By default the bundled training data are explained: one row per locus and class
    values = pd.read_csv(os.path.join(output_dir(tmp_path), "shap_values.csv"))
    assert len(values) == 3 * 2907
    assert sorted(values["class"].unique()) == ["Candidate", "Confident", "falsepositive"]
    importance = pd.read_csv(os.path.join(output_dir(tmp_path), "shap_importance.csv"))
    assert list(importance.columns) == ["feature", "name", "Candidate", "Confident", "falsepositive"]
    assert len(importance) == 18


def test_legacy_importance_option(small_model, tmp_path):
    files = visualize(tmp_path, "--model", small_model, "--legacy_importance")
    assert set(LEGACY_FILES) <= set(files)


def test_pdf_keeps_text_as_embedded_truetype(small_model, tmp_path):
    visualize(tmp_path, "--model", small_model)
    with open(os.path.join(output_dir(tmp_path), "shap_summary_Confident.pdf"), "rb") as f:
        pdf = f.read()
    assert b"/FontFile2" in pdf and b"/Type3" not in pdf


def test_svg_format_keeps_text_as_text(small_model, tmp_path):
    files = visualize(tmp_path, "--model", small_model, "--format", "svg")
    assert [f for f in files if f.endswith(".svg")] == [f + ".svg" for f in SHAP_FILES]
    with open(os.path.join(output_dir(tmp_path), "shap_summary_Candidate.svg")) as f:
        svg = f.read()
    assert re.search(r"<text[^>]*>[^<]*Five Prime Homogeneity", svg)


def test_data_option_explains_a_run_with_blank_overhangs(small_model, tmp_path):
    data = pd.read_csv(TRAINING_CSV).iloc[:, :-1]
    data.loc[:20, ["3' Overhang Bot", "3' Overhang Top"]] = float("nan")
    path = tmp_path / "fully_formatted_data.csv"
    data.to_csv(path, index=False)
    visualize(tmp_path, "--model", small_model, "--data", str(path))
    values = pd.read_csv(os.path.join(output_dir(tmp_path), "shap_values.csv"))
    assert len(values) == 3 * len(data)
    assert set(values["provisional_id"]) == set(data["provisional_id"])


def test_trees_option_writes_each_class_of_each_round(small_model, tmp_path):
    trees = [f for f in visualize(tmp_path, "--model", small_model, "--trees", "2") if f.startswith("tree_")]
    assert trees == [f"tree_round{r:04d}_{c}.png" for r in range(2)
                     for c in sorted(["Candidate", "Confident", "falsepositive"])]


def test_tree_labels_follow_the_model_file():
    from boosted_forest_visualization import tree_labels

    booster = xgboost.Booster()
    booster.load_model(os.path.join(PKG, "model_weights", "miRNA_model.json"))
    labels = tree_labels(booster)
    assert len(labels) == 3 * booster.num_boosted_rounds() == 2949
    assert labels[:4] == [(0, "Candidate"), (0, "Confident"), (0, "falsepositive"), (1, "Candidate")]
    assert labels[-1] == (982, "falsepositive")


def test_trees_are_dumped_once(small_model, tmp_path, monkeypatch):
    from boosted_forest_visualization import export_trees

    monkeypatch.setenv("PATH", ENV["PATH"])  # graphviz's dot lives in the conda env
    calls = []
    original = xgboost.Booster.get_dump
    monkeypatch.setattr(xgboost.Booster, "get_dump", lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    booster = xgboost.Booster()
    booster.load_model(small_model)
    paths = export_trees(booster, str(tmp_path), rounds=None)
    assert len(paths) == 9
    assert len(calls) == 1
