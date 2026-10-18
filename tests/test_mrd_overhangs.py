"""B3: signed 3' overhangs at the two ends of the mature-star duplex, from output.mrd.

Duplex rule (Tejus, 18 Oct 2026): only base pairs between the mature and the
star count, because Drosha and Dicer cut off the flanks and the loop. At the
Drosha end the outermost of these pairs is taken, at the Dicer end the
innermost, and the overhang is the number of unpaired nucleotides beyond it on
the strand whose 3' end is there, minus the number on the strand whose 5' end
is there: +2 for a canonical cut, 0 for a blunt end, negative for a 5'
overhang. Loci without star reads have no overhangs.
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


def structure(length, pairs):
    """Dot-bracket string with the given (i, j) base pairs, 1-based, i < j."""
    marks = ["."] * length
    for i, j in pairs:
        marks[i - 1], marks[j - 1] = "(", ")"
    return "".join(marks)


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
    # Mature 13-34. On a perfect helix this is the star's 3' end against 70, the
    # partner of the mature's 5' end (Drosha end), and the mature's 3' end (34)
    # against the partner of the star's 5' end (Dicer end; 51 pairs with 32).
    assert overhangs(locus((13, 34), star)) == (drosha, dicer)


@pytest.mark.parametrize("star, drosha, dicer", [
    ((13, 34), 2, 2),
    ((12, 34), 1, 2),
    ((13, 36), 2, 4),
    ((15, 32), 4, 0),
])
def test_mature_on_3p_arm(star, drosha, dicer):
    # Mature 51-72. On a perfect helix this is the mature's 3' end (72) against
    # the partner of the star's 5' end (Drosha end), and the star's 3' end
    # against 32, the partner of the mature's 5' end (Dicer end).
    assert overhangs(locus((51, 72), star)) == (drosha, dicer)


def test_unpaired_end_nucleotides_dangle():
    # The mature's 5' nucleotide (13) is unpaired. The outermost pair is 14-69:
    # 3 star nucleotides beyond it (70-72) minus 1 mature nucleotide (13).
    unpaired_5p = "." * 13 + "(" * 23 + "." * 10 + ")" * 23 + "." * 13
    assert overhangs(locus((13, 34), (51, 72), unpaired_5p)) == (2, 2)
    # 32-36 are unpaired. The innermost pair is 31-47: 3 mature nucleotides
    # beyond it (32-34) minus 1 star nucleotide (46).
    unpaired_3p = "." * 10 + "(" * 21 + "." * 15 + ")" * 21 + "." * 10
    assert overhangs(locus((13, 34), (46, 67), unpaired_3p)) == (2, 2)


# Star 21-42 on the 5' arm, mature 61-81 on the 3' arm, paired i with 100 - i
# from 21-79 to 38-62. Star 39-42 are unpaired and loop nucleotide 43 pairs with
# the mature's 5' end (61), as in Pomonella's QFTL02000135.1_4908.
LOOP_PAIR = structure(92, [(i, 100 - i) for i in range(21, 39)] + [(43, 61)])


def test_pairs_with_the_loop_do_not_count():
    # Innermost mature-star pair 38-62: 4 star nucleotides beyond it (39-42)
    # minus 1 mature nucleotide (61). miRDeep2's expected-star rule measures
    # from 61's partner in the loop (43) instead and gives -1.
    assert overhangs(locus((61, 81), (21, 42), LOOP_PAIR)) == (2, 3)


def test_same_duplex_whichever_strand_is_mature():
    # miRDeep2's rule gives (2, -1) with the mature on the 3' arm and (2, 3) with it on the 5' arm
    assert overhangs(locus((21, 42), (61, 81), LOOP_PAIR)) == overhangs(locus((61, 81), (21, 42), LOOP_PAIR))


def test_bulge_next_to_the_end():
    # As in Pomonella's QFTL02000007.1_25533: the mature's last nucleotide (82)
    # pairs with star nucleotide 24, then mature 77-81 bulge, then 25-76 to
    # 40-61. Drosha end: outermost pair 24-82, 0 mature nucleotides beyond it
    # minus 3 star nucleotides (21-23). miRDeep2's rule steps over the bulge,
    # treats it as paired and gives 2.
    bulge = structure(92, [(i, 101 - i) for i in range(25, 41)] + [(24, 82)])
    assert overhangs(locus((61, 82), (21, 42), bulge)) == (-3, 2)


def test_frayed_end_reads_as_blunt():
    # Known limit (METHODS_CHANGES.md, B3): mature 30-34 and star 49-53 are all
    # unpaired, 5 nucleotides beyond the innermost pair (29-54) on each strand,
    # which nets to 0, the same as a blunt end.
    frayed = structure(82, [(i, 83 - i) for i in range(11, 30)])
    assert overhangs(locus((13, 34), (49, 72), frayed)) == (2, 0)


def test_no_mature_star_pair_gives_no_overhangs():
    drosha, dicer = overhangs(locus((13, 34), (51, 72), "." * len(HAIRPIN)))
    assert math.isnan(drosha) and math.isnan(dicer)


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
# - Let-7-P2a1: the 5p arm's 5' nucleotide (31) is a bulge. Outermost pair
#   32-100: 2 nucleotides of the 3p arm beyond it minus 1 of the 5p arm.
# - Mir-145: outermost pair 31-89, and the 3p arm ends 1 nt beyond it.
# - Mir-21: innermost pair 51-69, and MirGeneDB's 5p arm (24 nt) ends 3 nt beyond it.
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
