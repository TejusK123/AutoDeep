"""D1: the saved model is the chosen configuration refit on all rows with balanced weights.

Before D1, tuning refit the best configuration on the first, unstratified 80%
split without the balanced class weights used to score it, and the -hp path
saved a model fit on a stratified 80% split. The number of trees is chosen by
early stopping instead of being tuned, and the search space lets max_depth go
past 5 and no longer tunes base_score, which has no effect under softmax.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xgboost
from sklearn.utils import class_weight

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")
AUTODEEP = os.path.join(PKG, "scripts", "AutoDeep.sh")
TRAINING_CSV = os.path.join(PKG, "training_data_pre", "original_data_Pomonella.csv")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
LABELS = ("-n", "-t", "labels.csv")


@pytest.fixture
def run_dir(tmp_path):
    """An AutoDeepRun folder with pipeline output and a labels file for it."""
    data = pd.read_csv(TRAINING_CSV)
    run = tmp_path / "AutoDeepRun"
    run.mkdir()
    data.iloc[:, :-1].to_csv(run / "fully_formatted_data.csv", index=False)
    data[["provisional_id"]].assign(label=data.iloc[:, -1]).to_csv(run / "labels.csv", index=False)
    return run


def train(run, name, *args):
    result = subprocess.run(["bash", AUTODEEP, "train", *args, "--out_dir", f"models/{name}"], cwd=run, env=ENV,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    out = run / "models" / name
    model = xgboost.XGBClassifier()
    model.load_model(str(out / "miRNA_model.json"))
    return json.loads((out / "config.json").read_text()), json.loads((out / "metrics.json").read_text()), model


def refit_on_all_rows(run, config, monkeypatch, tmp_path):
    """The chosen configuration fit on every training row, with balanced class weights."""
    from boosted_forest_training import load_training_set

    monkeypatch.chdir(run)
    features, targets, _ = load_training_set(["labels.csv"], True, str(tmp_path))
    model = xgboost.XGBClassifier(**config["hyperparameters"], random_state=config["seed"])
    model.fit(features, targets, sample_weight=class_weight.compute_sample_weight("balanced", targets))
    return model


def same_trees(a, b):
    return a.get_booster().get_dump() == b.get_booster().get_dump()


def test_manual_hyperparameters_model_is_refit_on_all_rows(run_dir, monkeypatch, tmp_path):
    (run_dir / "hp.txt").write_text("n_estimators 5\nmax_depth 2\nsubsample 0.5\n")
    config, _, model = train(run_dir, "hp", *LABELS, "-hp", "hp.txt", "--seed", "7")
    assert config["hyperparameters"]["n_estimators"] == 5  # given in the file, so used as is
    assert same_trees(model, refit_on_all_rows(run_dir, config, monkeypatch, tmp_path))


def test_tuned_model_is_refit_on_all_rows(run_dir, monkeypatch, tmp_path):
    config, _, model = train(run_dir, "tuned", *LABELS, "-r", "1", "--seed", "7")
    assert same_trees(model, refit_on_all_rows(run_dir, config, monkeypatch, tmp_path))


def test_tuning_chooses_the_number_of_trees_by_early_stopping(run_dir):
    from boosted_forest_training import MAX_ROUNDS

    config, metrics, model = train(run_dir, "tuned", *LABELS, "-r", "1", "--seed", "7")
    rounds = config["hyperparameters"]["n_estimators"]
    assert isinstance(rounds, int) and 1 <= rounds <= MAX_ROUNDS
    assert model.get_booster().num_boosted_rounds() == rounds
    assert config["final_fit"]["n_rows"] == 212
    assert {"mean_accuracy", "weighted_f1_score", "macro_f1_score"} <= set(metrics)


def test_manual_hyperparameters_are_scored_on_the_same_ten_splits(run_dir):
    (run_dir / "hp.txt").write_text("max_depth 2\neta 0.3\n")
    config, metrics, model = train(run_dir, "hp", *LABELS, "-hp", "hp.txt")
    # Without n_estimators in the file, early stopping chooses it
    assert config["hyperparameters"]["n_estimators"] == model.get_booster().num_boosted_rounds()
    assert {"mean_accuracy", "std_accuracy", "weighted_f1_score", "macro_f1_score"} <= set(metrics)


def test_score_config_stops_early_and_reports_mean_rounds():
    from boosted_forest_training import MAX_ROUNDS, score_config
    from features import encode_features, encode_labels

    data = pd.read_csv(TRAINING_CSV)
    features, targets = encode_features(data.iloc[:, 1:-1]), encode_labels(data.iloc[:, -1])
    scores = score_config({"objective": "multi:softprob", "max_depth": 2, "eta": 0.3}, features, targets, seed=1)
    assert 1 <= scores["n_estimators"] < MAX_ROUNDS
    fixed = score_config({"objective": "multi:softprob", "max_depth": 2, "n_estimators": 7}, features, targets, seed=1)
    assert fixed["n_estimators"] == 7
    assert 0 <= fixed["mean_accuracy"] <= 1


def test_search_space():
    from boosted_forest_training import tuning_search_space

    space = tuning_search_space()
    assert "base_score" not in space  # no effect under softmax
    assert "n_estimators" not in space  # chosen by early stopping
    # randint excludes its upper bound: max_depth 2-8
    assert (space["max_depth"].lower, space["max_depth"].upper) == (2, 9)
    assert (space["eta"].lower, space["eta"].upper) == (0.005, 0.3)
