"""A6: training is repeatable: one --seed fixes the data splits, XGBoost and the tuning sampler."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")
AUTODEEP = os.path.join(PKG, "scripts", "AutoDeep.sh")
TRAINING_CSV = os.path.join(PKG, "training_data_pre", "original_data_Pomonella.csv")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
# Row and column subsampling make XGBoost draw random numbers, so its seed matters too
HP = "n_estimators 5\nmax_depth 2\nsubsample 0.5\ncolsample_bytree 0.8\n"
LABELS = ("-n", "-t", "labels.csv")


@pytest.fixture
def run_dir(tmp_path):
    """An AutoDeepRun folder with pipeline output, a labels file for it and a hyperparameter file."""
    data = pd.read_csv(TRAINING_CSV)
    run = tmp_path / "AutoDeepRun"
    run.mkdir()
    data.iloc[:, :-1].to_csv(run / "fully_formatted_data.csv", index=False)
    data[["provisional_id"]].assign(label=data.iloc[:, -1]).to_csv(run / "labels.csv", index=False)
    (run / "hp.txt").write_text(HP)
    return run


def train(run, name, *args):
    """Train into models/<name>; return its config, metrics and model file contents."""
    result = subprocess.run(["bash", AUTODEEP, "train", *args, "--out_dir", f"models/{name}"], cwd=run, env=ENV,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    out = run / "models" / name
    return (json.loads((out / "config.json").read_text()), json.loads((out / "metrics.json").read_text()),
            (out / "miRNA_model.json").read_bytes())


def test_same_seed_gives_same_model(run_dir):
    config, metrics, model = train(run_dir, "a", *LABELS, "-hp", "hp.txt", "--seed", "7")
    _, metrics_again, model_again = train(run_dir, "b", *LABELS, "-hp", "hp.txt", "--seed", "7")
    assert metrics == metrics_again
    assert model == model_again
    assert config["seed"] == 7


def test_runs_without_seed_are_repeatable(run_dir):
    config, metrics, model = train(run_dir, "a", *LABELS, "-hp", "hp.txt")
    _, metrics_again, model_again = train(run_dir, "b", *LABELS, "-hp", "hp.txt")
    assert (metrics, model) == (metrics_again, model_again)
    assert config["seed"] == 42


def test_different_seeds_give_different_models(run_dir):
    *_, model = train(run_dir, "a", *LABELS, "-hp", "hp.txt", "--seed", "7")
    *_, other = train(run_dir, "b", *LABELS, "-hp", "hp.txt", "--seed", "8")
    assert model != other


def test_tuning_with_same_seed_is_repeatable(run_dir):
    config, metrics, model = train(run_dir, "a", *LABELS, "-r", "2", "--seed", "7")
    config_again, metrics_again, model_again = train(run_dir, "b", *LABELS, "-r", "2", "--seed", "7")
    assert config["hyperparameters"] == config_again["hyperparameters"]
    assert metrics == metrics_again
    assert model == model_again


@pytest.mark.parametrize("seed", ["-1", str(2 ** 31)])
def test_seed_out_of_range_is_rejected(run_dir, seed):
    result = subprocess.run(["bash", AUTODEEP, "train", *LABELS, "-hp", "hp.txt", "--seed", seed],
                            cwd=run_dir, env=ENV, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--seed must be" in result.stderr


def test_training_files_are_read_in_name_order(tmp_path, monkeypatch):
    # Directory listings come back in a different order on different machines,
    # and row order decides which loci a seeded split puts in the test set
    from boosted_forest_training import read_training_data

    for name in ["a.csv", "b.csv", "c.csv"]:
        pd.DataFrame({"provisional_id": [name]}).to_csv(tmp_path / name, index=False)
    monkeypatch.setattr(os, "walk", lambda path: iter([(str(tmp_path), [], ["c.csv", "a.csv", "b.csv"])]))
    assert list(read_training_data(str(tmp_path))["provisional_id"]) == ["a.csv", "b.csv", "c.csv"]


def suggestions(n, parallel, seed):
    """Configs the tuning sampler proposes for n trials when up to `parallel` run at once
    (the oldest finishes first). Drives the searcher by hand, without starting Ray."""
    from boosted_forest_training import tuning_search_alg, tuning_search_space

    search = tuning_search_alg(seed)
    search.set_search_properties("weighted_f1_score", "max", tuning_search_space())
    running, proposed = [], []
    while len(proposed) < n or running:
        while len(running) < parallel and len(proposed) < n:
            trial_id = f"trial{len(proposed)}"
            config = search.suggest(trial_id)
            running.append((trial_id, config))
            proposed.append(json.dumps(config, sort_keys=True))
        trial_id, config = running.pop(0)
        search.on_trial_complete(trial_id, {"weighted_f1_score": config["eta"] - abs(config["max_depth"] - 3)})
    return sorted(proposed)


def test_tuning_sampler_follows_the_seed():
    assert suggestions(3, 1, seed=7) == suggestions(3, 1, seed=7)
    assert suggestions(3, 1, seed=7) != suggestions(3, 1, seed=8)


def test_first_ten_tuning_rounds_do_not_depend_on_cpu_count():
    # Ray runs as many trials at once as there are CPUs. The sampler's first ten
    # proposals are seeded random draws, so the default -r 10 repeats on any machine.
    # Later proposals depend on which trials have finished (see METHODS_CHANGES.md, A6).
    assert suggestions(10, 1, seed=42) == suggestions(10, 10, seed=42)
