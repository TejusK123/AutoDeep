"""A5: tree plots are opt-in, dumped once, and labelled by round and class."""
import glob
import os
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


def visualize(cwd, *args):
    subprocess.run([sys.executable, VISUALIZE, *args], cwd=cwd, env=ENV, check=True, capture_output=True)
    (out,) = glob.glob(os.path.join(cwd, "tree_plots*"))
    return sorted(os.listdir(out))


def test_default_writes_importance_plots_only(small_model, tmp_path):
    files = visualize(tmp_path, "--model", small_model)
    assert files == ["feature_importance_cover.png", "feature_importance_gain.png", "feature_importance_weight.png"]


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
