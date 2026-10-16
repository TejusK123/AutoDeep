"""A1: inference must encode flags the same way whatever values the user's file holds."""
import os
import subprocess
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")
INFERENCE = os.path.join(PKG, "scripts", "boosted_forest_inference.py")
# The data the shipped model was trained on
TRAINING_CSV = os.path.join(PKG, "training_data_pre", "original_data_Human_fp_candidate.csv")

FLAGS = ["significant_randfold_p-value", "homologous_seed_in_miRBase", "mature_5'u_or_3'u"]


def run_inference(features, workdir):
    run_dir = os.path.join(workdir, "AutoDeepRun")
    os.makedirs(run_dir, exist_ok=True)
    features.to_csv(os.path.join(run_dir, "fully_formatted_data.csv"), index=False)
    subprocess.run([sys.executable, INFERENCE], cwd=workdir, check=True)
    return pd.read_csv(os.path.join(run_dir, "xgboost_judgement.csv"), dtype=str)


@pytest.fixture(scope="module")
def mixed_features():
    # Inference input has the training columns minus the label column
    return pd.read_csv(TRAINING_CSV).iloc[:, :-1]


@pytest.fixture(scope="module")
def mixed_output(mixed_features, tmp_path_factory):
    return run_inference(mixed_features, str(tmp_path_factory.mktemp("mixed")))


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("value_index", [0, 1])
def test_single_valued_flag_column(flag, value_index, mixed_features, mixed_output, tmp_path):
    value = sorted(mixed_features[flag].unique(), key=str)[value_index]
    subset = mixed_features[mixed_features[flag] == value].reset_index(drop=True)
    assert len(subset) > 0

    out = run_inference(subset, str(tmp_path))
    expected = mixed_output.set_index("provisional_id").loc[subset["provisional_id"]]

    assert (out[flag] == str(value)).all(), f"{flag}={value!r} written back as {sorted(out[flag].unique())}"
    assert list(out["prediction"]) == list(expected["prediction"])
