"""A3: hyperparameter files are parsed as data, never evaluated as code."""
import os
import subprocess
import sys

import pandas as pd
import pytest

from hyperparameters import load_xgboost_hyperparameters, read_hyperparameter_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")
AUTODEEP = os.path.join(PKG, "scripts", "AutoDeep.sh")
ALT = os.path.join(PKG, "scripts", "alt_model_test.py")
TRAINING_CSV = os.path.join(PKG, "training_data_pre", "original_data_Pomonella.csv")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
EVIL = "max_depth __import__('pathlib').Path('pwned').touch()\n"


def write(tmp_path, text, name="hp.txt"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


@pytest.fixture
def run_dir(tmp_path):
    data = pd.read_csv(TRAINING_CSV)
    run = tmp_path / "AutoDeepRun"
    run.mkdir()
    data.iloc[:, :-1].to_csv(run / "fully_formatted_data.csv", index=False)
    data[["provisional_id"]].assign(label=data.iloc[:, -1]).to_csv(run / "labels.csv", index=False)
    return run


def test_train_does_not_run_code_from_hyperparameter_file(run_dir):
    (run_dir / "evil.txt").write_text(EVIL)
    result = subprocess.run(["bash", AUTODEEP, "train", "-n", "-t", "labels.csv", "-hp", "evil.txt", "-nw",
                             "--out_dir", "models/evil"], cwd=run_dir, env=ENV, capture_output=True, text=True)
    assert not (run_dir / "pwned").exists()
    assert result.returncode != 0
    assert "max_depth" in result.stdout + result.stderr


def test_alt_models_do_not_run_code_from_hyperparameter_file(run_dir):
    (run_dir / "evil.txt").write_text(EVIL)
    result = subprocess.run([sys.executable, ALT, "-m", "svm", "-hp", "evil.txt", "-nw"],
                            cwd=run_dir, env=ENV, capture_output=True, text=True)
    assert not (run_dir / "pwned").exists()
    assert result.returncode != 0


@pytest.mark.parametrize("model, text", [
    ("svm", "C 0.5\nkernel 'rbf'\n"),
    ("knn", "n_neighbors 7\nweights 'distance'\n"),
    ("mlp", "hidden_layer_sizes (8, 4)\nmax_iter 50\n"),
])
def test_alt_models_still_take_their_hyperparameters(run_dir, model, text):
    (run_dir / "hp.txt").write_text(text)
    result = subprocess.run([sys.executable, ALT, "-m", model, "-hp", "hp.txt", "-nw"],
                            cwd=run_dir, env=ENV, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Manual-tuned Model Metrics" in result.stdout


def test_text_format_values_are_literals(tmp_path):
    path = write(tmp_path, "# comment\nkernel 'rbf'\n\nhidden_layer_sizes (100, 50)\nC 1e-2\nshrinking True\n")
    assert read_hyperparameter_file(path) == {
        "kernel": "rbf", "hidden_layer_sizes": (100, 50), "C": 0.01, "shrinking": True}


@pytest.mark.parametrize("text", [EVIL, "C rbf\n", "C open('x')\n", "max_depth\n"])
def test_text_format_rejects_non_literals(tmp_path, text):
    with pytest.raises(ValueError):
        read_hyperparameter_file(write(tmp_path, text))


def test_json_format(tmp_path):
    path = write(tmp_path, '{"max_depth": 3, "eta": 0.1, "n_estimators": 200}', "hp.json")
    assert load_xgboost_hyperparameters(path) == {"max_depth": 3, "eta": 0.1, "n_estimators": 200}


def test_invalid_json_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        read_hyperparameter_file(write(tmp_path, '{"max_depth": __import__("os")}', "hp.json"))


def test_config_json_from_a_training_run(tmp_path):
    path = write(tmp_path, '{"hyperparameters": {"objective": "multi:softprob", "max_depth": 4},'
                           ' "n_rows": 10}', "config.json")
    assert load_xgboost_hyperparameters(path) == {"objective": "multi:softprob", "max_depth": 4}


@pytest.mark.parametrize("text, key", [
    ("booster 'gblinear'\n", "booster"),
    ("eta 'fast'\n", "eta"),
    ("subsample True\n", "subsample"),
    ("max_depth 3.5\n", "max_depth"),
    ("n_estimators 0\n", "n_estimators"),
    ("objective 'binary:logistic'\n", "objective"),
])
def test_xgboost_values_are_validated(tmp_path, text, key):
    with pytest.raises(ValueError, match=key):
        load_xgboost_hyperparameters(write(tmp_path, text))
