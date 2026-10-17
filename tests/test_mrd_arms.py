"""B1: per-arm read counts from output.mrd equal miRDeep2's own counts in result.csv.

The tutorial run is in the repo, so that case runs in CI. The Pomonella case
needs a local miRDeep2 run and is skipped unless AUTODEEP_TEST_RUN points at a
directory holding mirdeep_Pomonella/ (result_*.csv, output.mrd).
"""
import glob
import os

import pytest

from mrd import arm_counts, parse_mrd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.environ.get("AUTODEEP_TEST_RUN")
RUNS = [
    pytest.param(os.path.join(REPO, "examples", "mirdeep2_tutorial", "mirdeep2_output"), 8, id="tutorial"),
    pytest.param(os.path.join(RUN or "", "mirdeep_Pomonella"), 212, id="pomonella",
                 marks=pytest.mark.skipif(not RUN, reason="set AUTODEEP_TEST_RUN to a local Pomonella run")),
]


def result_counts(path):
    """{id: {'mature', 'star', 'loop'}} from the novel and known miRNA tables of result.csv."""
    counts = {}
    header = None
    with open(path) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if cols[0] in ("provisional id", "tag id"):
                header = cols
            elif header and len(cols) == len(header):
                row = dict(zip(header, cols))
                counts[cols[0]] = {arm: int(row[f"{arm} read count"]) for arm in ("mature", "star", "loop")}
            else:
                header = None
    return counts


@pytest.mark.parametrize("run_dir, n_loci", RUNS)
def test_arm_counts_match_result_csv(run_dir, n_loci):
    (result,) = glob.glob(os.path.join(run_dir, "result_*.csv"))
    expected = result_counts(result)
    assert len(expected) == n_loci
    loci = parse_mrd(os.path.join(run_dir, "output.mrd"))
    for pid, counts in expected.items():
        assert arm_counts(loci[pid]) == counts, pid
