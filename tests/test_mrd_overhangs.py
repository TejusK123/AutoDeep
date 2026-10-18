"""B3: signed 3' overhangs at the two ends of the mature-star duplex, from output.mrd.

At each end, the overhang is how far one strand's 3' end extends past the
partner of the other strand's 5' end: +2 for a canonical Drosha or Dicer cut,
0 for a blunt end, negative when the overhang is on the 5' side. As in
miRDeep2 (find_star in miRDeep2_core_algorithm.pl), the mature's position and
the precursor structure give the star expected for 2-nt overhangs, and the
overhang is 2 plus the observed star's offset from it. Loci without star reads
have no overhangs.
"""
import glob
import math
import os

import pandas as pd
import pytest

from mrd import Locus, overhangs, parse_mrd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.environ.get("AUTODEEP_TEST_RUN")

# A perfect hairpin: 5' flank 1-10, 5' arm 11-36, loop 37-46, 3' arm 47-72 and
# 3' flank 73-82, so position i pairs with 83 - i.
HAIRPIN = "." * 10 + "(" * 26 + "." * 10 + ")" * 26 + "." * 10


def obs_line(length, mature, star):
    """miRDeep2's obs string (flank f, mature M, loop l, star S) for (beg, end) positions, 1-based."""
    marks = ["f"] * length
    (_, first_end), (second_beg, _) = sorted([mature, star])
    marks[first_end:second_beg - 1] = "l" * (second_beg - 1 - first_end)
    for (beg, end), mark in ((mature, "M"), (star, "S")):
        marks[beg - 1:end] = mark * (end - beg + 1)
    return "".join(marks)


def locus(mature, star, structure=HAIRPIN):
    line = obs_line(len(structure), mature, star)
    return Locus(exp=line, reads=[], obs=line, pri_struct=structure)


@pytest.mark.parametrize("star, drosha, dicer", [
    ((51, 72), 2, 2),
    ((52, 73), 3, 3),
    ((50, 71), 1, 1),
    ((49, 70), 0, 0),
    ((48, 69), -1, -1),
    ((51, 69), -1, 2),
    ((53, 72), 2, 4),
])
def test_mature_on_5p_arm(star, drosha, dicer):
    # Mature 13-34. Drosha end: the star's 3' end against 70, the partner of the
    # mature's 5' end. Dicer end: the mature's 3' end (34) against the partner
    # of the star's 5' end (51 pairs with 32).
    assert overhangs(locus((13, 34), star)) == (drosha, dicer)


@pytest.mark.parametrize("star, drosha, dicer", [
    ((13, 34), 2, 2),
    ((12, 34), 1, 2),
    ((13, 36), 2, 4),
    ((15, 32), 4, 0),
])
def test_mature_on_3p_arm(star, drosha, dicer):
    # Mature 51-72. Drosha end: the mature's 3' end (72) against the partner of
    # the star's 5' end. Dicer end: the star's 3' end against 32, the partner of
    # the mature's 5' end.
    assert overhangs(locus((51, 72), star)) == (drosha, dicer)


def test_unpaired_end_counts_as_continuing_the_helix():
    # As in miRDeep2. The mature's 5' nucleotide (13) is unpaired, so its partner
    # is taken to be one past 69, the partner of 14.
    structure = "." * 13 + "(" * 23 + "." * 10 + ")" * 23 + "." * 13
    assert overhangs(locus((13, 34), (51, 72), structure)) == (2, 2)
    # 32-36 are unpaired, so 32's partner is taken to be 46, one before 47, the partner of 31
    structure = "." * 10 + "(" * 21 + "." * 15 + ")" * 21 + "." * 10
    assert overhangs(locus((13, 34), (46, 67), structure)) == (2, 2)


def test_no_star_reads_gives_no_overhangs():
    # Without star reads miRDeep2 writes no obs line. Its exp star is then a
    # prediction that assumes 2-nt overhangs, so it says nothing about the locus.
    line = obs_line(len(HAIRPIN), (13, 34), (51, 72))
    drosha, dicer = overhangs(Locus(exp=line, reads=[], pri_struct=HAIRPIN))
    assert math.isnan(drosha) and math.isnan(dicer)


def test_missing_structure_raises():
    line = obs_line(len(HAIRPIN), (13, 34), (51, 72))
    with pytest.raises(ValueError):
        overhangs(Locus(exp=line, reads=[], obs=line))


# Ten MirGeneDB 3.0 precursors chosen in advance, with their ViennaRNA 2.7.0
# folds (tests/data/mirgenedb_hsa_ten.csv). Seven give 2 at both ends. The
# exceptions follow from their folds:
# - Let-7-P2a1: the 5p arm's 5' nucleotide is a bulge, so miRDeep2's rule takes
#   its partner from the next pair in, one short of the 3p arm's end.
# - Mir-145: the 3p arm ends 1 nt past the partner of the 5p arm's 5' end.
# - Mir-21: MirGeneDB's 5p arm (24 nt) ends 3 nt past the partner of the 3p arm's 5' end.
MIRGENEDB_EXCEPTIONS = {"Hsa-Let-7-P2a1": (1, 2), "Hsa-Mir-145": (1, 2), "Hsa-Mir-21": (2, 3)}
MIRGENEDB = pd.read_csv(os.path.join(REPO, "tests", "data", "mirgenedb_hsa_ten.csv"), comment="#")


@pytest.mark.parametrize("row", MIRGENEDB.to_dict("records"), ids=list(MIRGENEDB["gene"]))
def test_mirgenedb_precursors(row):
    arm5p = (row["arm5p_beg"], row["arm5p_end"])
    arm3p = (row["arm3p_beg"], row["arm3p_end"])
    mature, star = (arm5p, arm3p) if row["mature_arm"] == "5p" else (arm3p, arm5p)
    assert overhangs(locus(mature, star, row["structure"])) == MIRGENEDB_EXCEPTIONS.get(row["gene"], (2, 2))


def test_mirgenedb_fixture_has_ten():
    assert len(MIRGENEDB) == 10
    assert set(MIRGENEDB_EXCEPTIONS) <= set(MIRGENEDB["gene"])


# Real miRDeep2 runs. The tutorial run is in the repo, so that case runs in CI;
# the Pomonella case is skipped unless AUTODEEP_TEST_RUN points at a local run.
RUNS = [
    pytest.param(os.path.join(REPO, "examples", "mirdeep2_tutorial", "mirdeep2_output"), 8, id="tutorial"),
    pytest.param(os.path.join(RUN or "", "mirdeep_Pomonella"), 212, id="pomonella",
                 marks=pytest.mark.skipif(not RUN, reason="set AUTODEEP_TEST_RUN to a local Pomonella run")),
]


def result_rows(path):
    """{id: row} from the novel and known miRNA tables of result.csv."""
    rows = {}
    header = None
    with open(path) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if cols[0] in ("provisional id", "tag id"):
                header = cols
            elif header and len(cols) == len(header):
                rows[cols[0]] = dict(zip(header, cols))
            else:
                header = None
    return rows


@pytest.mark.parametrize("run_dir, n_loci", RUNS)
def test_observed_star_is_result_csv_star(run_dir, n_loci):
    # The star the overhangs are measured on is the one miRDeep2 reports
    (result,) = glob.glob(os.path.join(run_dir, "result_*.csv"))
    rows = result_rows(result)
    assert len(rows) == n_loci
    loci = parse_mrd(os.path.join(run_dir, "output.mrd"))
    for pid, row in rows.items():
        locus = loci[pid]
        marks = locus.obs or locus.exp
        star = "".join(nt for nt, mark in zip(locus.pri_seq, marks) if mark == "S")
        assert star == row["consensus star sequence"], pid


@pytest.mark.parametrize("run_dir, n_loci", RUNS)
def test_overhangs_only_with_star_reads(run_dir, n_loci):
    (result,) = glob.glob(os.path.join(run_dir, "result_*.csv"))
    rows = result_rows(result)
    loci = parse_mrd(os.path.join(run_dir, "output.mrd"))
    for pid, row in rows.items():
        has_star = int(row["star read count"]) > 0
        assert [math.isnan(x) for x in overhangs(loci[pid])] == [not has_star] * 2, pid
