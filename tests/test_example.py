"""A7: AutoDeep infer on miRDeep2's tutorial run gives the expected output.

examples/mirdeep2_tutorial/mirdeep2_output/ holds result_*.csv and output.mrd from
miRDeep2's own tutorial (see examples/mirdeep2_tutorial/README.md), and
expected/xgboost_judgement.csv is AutoDeep's output on it with the bundled model.
Unlike the Pomonella end-to-end tests, this one needs no local data, so it runs in CI.

Each test runs twice: on the run as it is, and with a comma added to the contig
name in result_*.csv and output.mrd (A8). Contig names can contain commas (NCBI
RefSeq names often do) and AutoDeep removes them, so the output must not change.
"""
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTODEEP = os.path.join(REPO, "AutoDeep", "scripts", "AutoDeep.sh")
EXAMPLE = os.path.join(REPO, "examples", "mirdeep2_tutorial")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])


@pytest.fixture(scope="module", params=["as_is", "commas_in_ids"])
def outputs(request, tmp_path_factory):
    # init_data.py rewrites result_*.csv in place, so run on a copy
    run_dir = tmp_path_factory.mktemp("mirdeep2_output")
    inputs = glob.glob(os.path.join(EXAMPLE, "mirdeep2_output", "*"))
    assert inputs, "the example's miRDeep2 output is missing"
    for path in inputs:
        shutil.copy(path, run_dir)
    if request.param == "commas_in_ids":
        for path in run_dir.iterdir():
            text = path.read_text()
            assert "chrII:" in text
            path.write_text(text.replace("chrII:", "chrII,:"))
    work = tmp_path_factory.mktemp("work")
    result = subprocess.run(["bash", AUTODEEP, "infer", str(run_dir)], cwd=work, env=ENV,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return (pd.read_csv(work / "AutoDeepRun" / "xgboost_judgement.csv"),
            pd.read_csv(os.path.join(EXAMPLE, "expected", "xgboost_judgement.csv")))


def test_example_gives_expected_labels(outputs):
    new, expected = outputs
    assert list(new.columns) == list(expected.columns)
    assert list(new["provisional_id"]) == list(expected["provisional_id"])
    assert list(new["prediction"]) == list(expected["prediction"])


SHAP_VALUES = [f"shap_top{k}_value" for k in (1, 2, 3)]


def test_example_gives_expected_features(outputs):
    # When a feature definition changes on purpose (Phase B), regenerate the expected file
    new, expected = outputs
    features = [c for c in expected.columns if c != "prediction" and not c.endswith("_Prob") and c not in SHAP_VALUES]
    pd.testing.assert_frame_equal(new[features], expected[features], check_exact=False, atol=1e-9, rtol=0)


def test_example_gives_expected_probabilities(outputs):
    new, expected = outputs
    probabilities = [c for c in expected.columns if c.endswith("_Prob")]
    assert probabilities
    # XGBoost's arithmetic differs slightly between x86 and arm64
    assert (new[probabilities] - expected[probabilities]).abs().max().max() < 1e-6


def test_example_gives_expected_top_contributions(outputs):
    # C4: the names are compared with the features above; the values, like the probabilities, nearly
    new, expected = outputs
    assert (new[SHAP_VALUES] - expected[SHAP_VALUES]).abs().max().max() < 1e-5
