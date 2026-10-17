"""A4: alt_model_test.py trains SVM, KNN and MLP models end to end, on the same data as AutoDeep train,
and writes only to its run folder."""
import hashlib
import json
import os
import subprocess
import sys

import joblib
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")
ALT = os.path.join(PKG, "scripts", "alt_model_test.py")
TRAINING_CSV = os.path.join(PKG, "training_data_pre", "original_data_Pomonella.csv")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
MLP_HP = "hidden_layer_sizes (8,)\nmax_iter 50\n"


def package_hashes():
    hashes = {}
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                hashes[os.path.relpath(path, PKG)] = hashlib.sha256(f.read()).hexdigest()
    return hashes


@pytest.fixture
def run_dir(tmp_path):
    """An AutoDeepRun folder with pipeline output, a labels file for it and hyperparameter files."""
    data = pd.read_csv(TRAINING_CSV)
    run = tmp_path / "AutoDeepRun"
    run.mkdir()
    data.iloc[:, :-1].to_csv(run / "fully_formatted_data.csv", index=False)
    data[["provisional_id"]].assign(label=data.iloc[:, -1]).to_csv(run / "labels.csv", index=False)
    (run / "knn.txt").write_text("n_neighbors 7\n")
    (run / "mlp.txt").write_text(MLP_HP)
    return run


def alt(run, *args):
    return subprocess.run([sys.executable, ALT, *args], cwd=run, env=ENV, capture_output=True, text=True)


def outputs(run, name):
    out = run / "models" / name
    return out, json.loads((out / "config.json").read_text()), json.loads((out / "metrics.json").read_text())


def pipeline_features(run):
    from features import encode_features

    return encode_features(pd.read_csv(run / "fully_formatted_data.csv").iloc[:, 1:])


@pytest.mark.parametrize("model, estimator", [("svm", SVC), ("knn", KNeighborsClassifier), ("mlp", MLPClassifier)])
def test_tuning_saves_a_fitted_model_of_the_requested_type(run_dir, model, estimator):
    result = alt(run_dir, "-m", model, "-n", "-t", "labels.csv", "-r", "1", "--out_dir", "models/run")
    assert result.returncode == 0, result.stdout + result.stderr
    out, config, metrics = outputs(run_dir, "run")
    fitted = joblib.load(out / f"{model}_model.pkl")
    assert isinstance(fitted, Pipeline) and isinstance(fitted[-1], estimator)
    assert len(fitted.predict(pipeline_features(run_dir))) == 212
    assert config["model"] == model
    assert "weighted_f1_score" in metrics
    assert (out / "model_training_log.csv").exists()


def test_manual_hyperparameters_save_the_fitted_model(run_dir):
    result = alt(run_dir, "-m", "knn", "-n", "-t", "labels.csv", "-hp", "knn.txt", "--out_dir", "models/hp")
    assert result.returncode == 0, result.stdout + result.stderr
    out, config, metrics = outputs(run_dir, "hp")
    fitted = joblib.load(out / "knn_model.pkl")
    assert isinstance(fitted, Pipeline) and fitted[-1].n_neighbors == 7
    assert config["hyperparameters"] == {"n_neighbors": 7}
    assert {"accuracy", "weighted_f1_score", "macro_f1_score"} <= set(metrics)


def test_training_leaves_package_untouched(run_dir):
    before = package_hashes()
    result = alt(run_dir, "-m", "knn", "-t", "labels.csv", "-hp", "knn.txt", "--out_dir", "models/pkg")
    assert result.returncode == 0, result.stdout + result.stderr
    assert package_hashes() == before
    assert {"knn_model.pkl", "config.json", "metrics.json", "user_data.csv"} <= set(os.listdir(run_dir / "models" / "pkg"))


def test_trains_on_the_same_data_as_autodeep_train(run_dir):
    result = alt(run_dir, "-m", "knn", "-t", "labels.csv", "-hp", "knn.txt", "-nw", "--out_dir", "models/data")
    assert result.returncode == 0, result.stdout + result.stderr
    _, config, _ = outputs(run_dir, "data")
    assert os.path.basename(config["training_data"]) == "training_data_pre"
    # The Pomonella labels repeat rows already in the bundled data (as in test_training_outputs.py)
    assert config["n_rows"] == 2907


def test_mismatched_user_columns_are_rejected(run_dir):
    data = pd.read_csv(run_dir / "fully_formatted_data.csv")
    data.insert(1, "extra_feature", 0)
    data.to_csv(run_dir / "fully_formatted_data.csv", index=False)
    result = alt(run_dir, "-m", "knn", "-t", "labels.csv", "-hp", "knn.txt", "-nw", "--out_dir", "models/bad")
    assert result.returncode != 0
    assert "do not match" in result.stdout + result.stderr


def test_no_weights_skips_model_only(run_dir):
    result = alt(run_dir, "-m", "knn", "-n", "-t", "labels.csv", "-hp", "knn.txt", "-nw", "--out_dir", "models/nw")
    assert result.returncode == 0, result.stdout + result.stderr
    files = set(os.listdir(run_dir / "models" / "nw"))
    assert "knn_model.pkl" not in files and "config.json" in files


def test_same_seed_gives_same_model(run_dir):
    args = ("-m", "mlp", "-n", "-t", "labels.csv", "-hp", "mlp.txt")
    runs = {}
    for name, seed in [("a", "7"), ("b", "7"), ("c", "8")]:
        result = alt(run_dir, *args, "--seed", seed, "--out_dir", f"models/{name}")
        assert result.returncode == 0, result.stdout + result.stderr
        out, config, metrics = outputs(run_dir, name)
        probabilities = joblib.load(out / "mlp_model.pkl").predict_proba(pipeline_features(run_dir))
        runs[name] = (config, metrics, probabilities)
    assert runs["a"][1] == runs["b"][1]
    assert (runs["a"][2] == runs["b"][2]).all()
    assert runs["a"][0]["seed"] == 7
    assert not (runs["a"][2] == runs["c"][2]).all()
