"""A2: training writes only to its run folder, never inside the package."""
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd
import pytest
import xgboost

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")
AUTODEEP = os.path.join(PKG, "scripts", "AutoDeep.sh")
INFERENCE = os.path.join(PKG, "scripts", "boosted_forest_inference.py")
TRAINING_CSV = os.path.join(PKG, "training_data_pre", "original_data_Pomonella.csv")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])


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
    """An AutoDeepRun folder with pipeline output and a labels file for it."""
    data = pd.read_csv(TRAINING_CSV)
    run = tmp_path / "AutoDeepRun"
    run.mkdir()
    data.iloc[:, :-1].to_csv(run / "fully_formatted_data.csv", index=False)
    data[["provisional_id"]].assign(label=data.iloc[:, -1]).to_csv(run / "labels.csv", index=False)
    (run / "hp.txt").write_text("n_estimators 5\nmax_depth 2\n")
    return run


def train(run, *args):
    return subprocess.run(["bash", AUTODEEP, "train", *args], cwd=run, env=ENV,
                          capture_output=True, text=True)


def test_training_with_labels_leaves_package_untouched(run_dir):
    before = package_hashes()
    result = train(run_dir, "-n", "-t", "labels.csv", "-hp", "hp.txt", "--out_dir", "models/run1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert package_hashes() == before

    out = run_dir / "models" / "run1"
    assert {"miRNA_model.json", "config.json", "metrics.json", "user_data.csv"} <= set(os.listdir(out))
    config = json.loads((out / "config.json").read_text())
    assert config["hyperparameters"] == {"n_estimators": 5, "max_depth": 2}
    assert config["n_rows"] == 212
    assert "accuracy" in json.loads((out / "metrics.json").read_text())


def test_tuning_leaves_package_untouched(run_dir):
    before = package_hashes()
    result = train(run_dir, "-r", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert package_hashes() == before

    (out,) = os.listdir(run_dir / "models")  # default: models/<timestamp>/
    files = set(os.listdir(run_dir / "models" / out))
    assert {"miRNA_model.json", "config.json", "metrics.json", "training_log.csv"} <= files


def test_no_weights_skips_model_only(run_dir):
    result = train(run_dir, "-n", "-t", "labels.csv", "-hp", "hp.txt", "-nw", "--out_dir", "models/nw")
    assert result.returncode == 0, result.stdout + result.stderr
    files = set(os.listdir(run_dir / "models" / "nw"))
    assert "miRNA_model.json" not in files and "config.json" in files


def test_existing_out_dir_is_not_overwritten(run_dir):
    (run_dir / "models" / "taken").mkdir(parents=True)
    result = train(run_dir, "-n", "-t", "labels.csv", "-hp", "hp.txt", "--out_dir", "models/taken")
    assert result.returncode != 0
    assert os.listdir(run_dir / "models" / "taken") == []


def test_user_data_from_earlier_run_can_be_reused(run_dir):
    assert train(run_dir, "-n", "-t", "labels.csv", "-hp", "hp.txt", "--out_dir", "models/a").returncode == 0
    result = train(run_dir, "-n", "-t", "models/a/user_data.csv", "labels.csv", "-hp", "hp.txt", "--out_dir", "models/b")
    assert result.returncode == 0, result.stdout + result.stderr
    # The same 212 loci twice: duplicates are dropped
    assert json.loads((run_dir / "models" / "b" / "config.json").read_text())["n_rows"] == 212


@pytest.fixture
def small_model(tmp_path):
    from features import encode_features, encode_labels

    data = pd.read_csv(TRAINING_CSV)
    model = xgboost.XGBClassifier(n_estimators=3, max_depth=2)
    model.fit(encode_features(data.iloc[:, 1:-1]), encode_labels(data.iloc[:, -1]))
    path = tmp_path / "small_model.json"
    model.save_model(path)
    return model, str(path), data


def test_inference_uses_model_option(small_model, tmp_path):
    from features import encode_features

    model, path, data = small_model
    work = tmp_path / "work"
    (work / "AutoDeepRun").mkdir(parents=True)
    data.iloc[:, :-1].to_csv(work / "AutoDeepRun" / "fully_formatted_data.csv", index=False)
    subprocess.run([sys.executable, INFERENCE, "--model", path], cwd=work, env=ENV, check=True)

    out = pd.read_csv(work / "AutoDeepRun" / "xgboost_judgement.csv")
    expected = model.predict_proba(encode_features(data.iloc[:, 1:-1]))[:, 1]
    assert (out["Confident_Prob"] - expected).abs().max() < 1e-6


def test_inference_missing_model_is_an_error(tmp_path):
    (tmp_path / "AutoDeepRun").mkdir()
    pd.read_csv(TRAINING_CSV).iloc[:, :-1].to_csv(tmp_path / "AutoDeepRun" / "fully_formatted_data.csv", index=False)
    result = subprocess.run([sys.executable, INFERENCE, "--model", "nope.json"], cwd=tmp_path, env=ENV,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "nope.json" in result.stderr
