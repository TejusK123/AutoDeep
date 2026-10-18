"""B1: features from output.mrd match the pypdf-era training data on a real run.

B3 redefined the 3' overhangs, so those two features differ from the training
data until B8 rebuilds it; the pipeline must write the values from output.mrd.

Needs a local miRDeep2 run, so it is skipped unless AUTODEEP_TEST_RUN points at
a directory holding mirdeep_Pomonella/ (result_*.csv, output.mrd).
"""
import glob
import os
import shutil
import subprocess

import pandas as pd
import pytest
import xgboost

from mrd import five_prime_features, overhangs, parse_mrd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")
TRAINING_CSV = os.path.join(PKG, "training_data_pre", "original_data_Pomonella.csv")
RUN = os.environ.get("AUTODEEP_TEST_RUN")

pytestmark = pytest.mark.skipif(not RUN, reason="set AUTODEEP_TEST_RUN to a local Pomonella run")

HOMOGENEITY = "five_prime_process_signal_selected_row_count_sum_over_mature_count_sum"
OVERHANGS = ["3' Overhang Bot", "3' Overhang Top"]


@pytest.fixture(scope="module")
def mrd_overhangs():
    """{id: (Drosha end, Dicer end)} from the run's output.mrd (B3)."""
    loci = parse_mrd(os.path.join(RUN, "mirdeep_Pomonella", "output.mrd"))
    return {pid: overhangs(locus) for pid, locus in loci.items()}


@pytest.fixture(scope="module")
def training():
    return pd.read_csv(TRAINING_CSV).set_index("provisional_id")


def test_mrd_features_match_training_data(training):
    loci = parse_mrd(os.path.join(RUN, "mirdeep_Pomonella", "output.mrd"))
    for pid, row in training.iterrows():
        homogeneity, num_locis = five_prime_features(loci[pid].reads)
        assert homogeneity == pytest.approx(row[HOMOGENEITY], abs=1e-9), pid
        assert num_locis == row["num_locis"], pid


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    # init_data.py rewrites result_*.csv in place, so run on a copy
    src = os.path.join(RUN, "mirdeep_Pomonella")
    run_dir = tmp_path_factory.mktemp("mirdeep_Pomonella")
    for path in glob.glob(os.path.join(src, "result_*.csv")) + [os.path.join(src, "output.mrd")]:
        shutil.copy(path, run_dir)
    work = tmp_path_factory.mktemp("work")
    subprocess.run(["bash", os.path.join(PKG, "scripts", "AutoDeep.sh"), "infer", str(run_dir)],
                   cwd=work, check=True)
    return os.path.join(work, "AutoDeepRun")


def test_pipeline_features_match_training_data(pipeline_run, training):
    model = xgboost.XGBClassifier()
    model.load_model(os.path.join(PKG, "model_weights", "miRNA_model.json"))
    names = [name for name in model.get_booster().feature_names if name not in OVERHANGS]

    out = pd.read_csv(os.path.join(pipeline_run, "fully_formatted_data.csv")).set_index("provisional_id")
    assert set(names) <= set(out.columns)
    assert sorted(out.index) == sorted(training.index)
    pd.testing.assert_frame_equal(out.loc[training.index, names], training[names],
                                  check_dtype=False, check_exact=False, atol=1e-9, rtol=0)


def test_pipeline_overhangs_come_from_output_mrd(pipeline_run, training, mrd_overhangs):
    out = pd.read_csv(os.path.join(pipeline_run, "fully_formatted_data.csv")).set_index("provisional_id")
    expected = pd.DataFrame([mrd_overhangs[pid] for pid in training.index], index=training.index, columns=OVERHANGS)
    # Blank (no star reads) must stay blank
    pd.testing.assert_frame_equal(out.loc[training.index, OVERHANGS], expected, check_dtype=False)


def test_pipeline_predictions_match_training_features(pipeline_run, training, mrd_overhangs):
    # The run's stored xgboost_judgement.csv came from an older model, so compare
    # with the shipped model's predictions on the pypdf-era training features,
    # with the overhangs from output.mrd (B3)
    from features import decode_labels, encode_features

    model = xgboost.XGBClassifier()
    model.load_model(os.path.join(PKG, "model_weights", "miRNA_model.json"))
    names = model.get_booster().feature_names
    features = training.copy()
    features[OVERHANGS] = [mrd_overhangs[pid] for pid in features.index]
    expected = decode_labels(model.predict(encode_features(features)[names].to_numpy()))

    new = pd.read_csv(os.path.join(pipeline_run, "xgboost_judgement.csv")).set_index("provisional_id")
    assert list(new.loc[training.index, "prediction"]) == list(expected)
