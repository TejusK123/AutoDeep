"""A8: AutoDeep infer runs on a miRDeep2 run whose locus IDs contain commas.

The S. frugiperda (FAW) run's contig names contain commas. init_data.py removes
them from result_*.csv (this copy of the run was already rewritten that way),
while output.mrd keeps them. The pipeline must still find every locus and give
the 5' features of the training rows made from this run.

Needs the local run, so it is skipped unless AUTODEEP_TEST_RUN points at a
directory holding mirDeep_FAW/ (result_*.csv, output.mrd).
"""
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "AutoDeep")
TRAINING_CSV = os.path.join(PKG, "training_data_pre", "original_data_FAW.csv")
RUN = os.environ.get("AUTODEEP_TEST_RUN")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])

pytestmark = pytest.mark.skipif(not RUN, reason="set AUTODEEP_TEST_RUN to the local testing folder")

FIVE_PRIME = ["five_prime_process_signal_selected_row_count_sum_over_mature_count_sum", "num_locis"]


def test_faw_run_gives_training_5p_features(tmp_path):
    # init_data.py rewrites result_*.csv in place, so run on a copy
    src = os.path.join(RUN, "mirDeep_FAW")
    run_dir = tmp_path / "mirDeep_FAW"
    run_dir.mkdir()
    for path in glob.glob(os.path.join(src, "result_*.csv")) + [os.path.join(src, "output.mrd")]:
        shutil.copy(path, run_dir)
    with open(run_dir / "output.mrd") as f:
        assert any(line.startswith(">") and "," in line for line in f)
    work = tmp_path / "work"
    work.mkdir()
    result = subprocess.run(["bash", os.path.join(PKG, "scripts", "AutoDeep.sh"), "infer", str(run_dir)],
                            cwd=work, env=ENV, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    out = pd.read_csv(work / "AutoDeepRun" / "fully_formatted_data.csv").set_index("provisional_id")
    training = pd.read_csv(TRAINING_CSV).set_index("provisional_id")
    assert len(training) == 145
    assert sorted(out.index) == sorted(training.index)
    pd.testing.assert_frame_equal(out.loc[training.index, FIVE_PRIME], training[FIVE_PRIME],
                                  check_dtype=False, check_exact=False, atol=1e-9, rtol=0)
