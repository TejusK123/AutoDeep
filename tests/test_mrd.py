"""B1: 5' homogeneity, num_locis and per-arm read counts from miRDeep2's output.mrd.

B2 redefined 5' homogeneity as the unbiased Simpson index of the mature-arm reads' 5' starts.
"""
import math

import pytest

from mrd import Locus, arm_counts, five_prime_features, parse_mrd

MRD = """\
>locus_1
score total\t\t      12.3
total read count\t        31
exp                           ffffSSSSllllMMMMff
obs                           ffffSSSSllllMMMMff
pri_seq                       acguagcuacguagcuac
pri_struct                    ..((((....))))....
seq_1_x10                     ....agcu..........\t0
seq_2_x5                      ....agc...........
seq_3_x3                      ....Cgcu..........\t1
seq_4_x2                      ......cuag........
seq_5_x4                      a.................
seq_6_x7                      ..............................ac


>locus_2
exp                           ffffMMMM
seq_7_x6                      ..ac....
seq_8_x6                      ....uagc
seq_9_x1                      ..Ac....\t1

"""


@pytest.fixture
def loci(tmp_path):
    path = tmp_path / "output.mrd"
    path.write_text(MRD)
    return parse_mrd(str(path))


def test_parse_keeps_only_reads_in_file_order(loci):
    assert list(loci) == ["locus_1", "locus_2"]
    assert [count for _, count in loci["locus_1"].reads] == [10, 5, 3, 2, 4, 7]
    assert loci["locus_1"].reads[2] == ("....Cgcu..........", 3)


def test_parse_keeps_exp_string(loci):
    assert loci["locus_1"].exp == "ffffSSSSllllMMMMff"
    assert loci["locus_2"].exp == "ffffMMMM"


def test_parse_keeps_obs_and_precursor(loci):
    assert loci["locus_1"].obs == "ffffSSSSllllMMMMff"
    assert loci["locus_1"].pri_seq == "acguagcuacguagcuac"
    assert loci["locus_1"].pri_struct == "..((((....))))...."
    # No star reads: miRDeep2 writes no obs line
    assert loci["locus_2"].obs is None
    assert loci["locus_2"].pri_struct is None


def test_parse_removes_commas_from_ids(tmp_path):
    # A8: init_data.py removes every comma from result_*.csv, so the IDs from
    # output.mrd must lose theirs too, or no locus is found
    path = tmp_path / "output.mrd"
    path.write_text(MRD.replace(">locus_1", ">chromosome_8,_ZJU_Sfru_1.0,_12098"))
    loci = parse_mrd(str(path))
    assert list(loci) == ["chromosome_8_ZJU_Sfru_1.0_12098", "locus_2"]
    assert [count for _, count in loci["chromosome_8_ZJU_Sfru_1.0_12098"].reads] == [10, 5, 3, 2, 4, 7]


def test_parse_rejects_ids_equal_without_commas(tmp_path):
    path = tmp_path / "output.mrd"
    path.write_text(MRD.replace(">locus_1", ">locus,_2"))
    with pytest.raises(ValueError, match="locus_2"):
        parse_mrd(str(path))


def test_homogeneity_counts_starts_whatever_the_first_base(loci):
    # All three reads are on the mature arm (5-8, allowing starts from 3). Read 9
    # starts at 3 like read 7 but with a mismatched first base: the same start.
    homogeneity, num_locis = five_prime_features(loci["locus_2"])
    assert homogeneity == pytest.approx((7 * 6 + 6 * 5) / (13 * 12))
    assert num_locis == 3


def test_homogeneity_blank_without_two_mature_reads(loci):
    # locus_1's reads are all on the star, the loop or past the precursor
    homogeneity, num_locis = five_prime_features(loci["locus_1"])
    assert math.isnan(homogeneity)
    assert num_locis == 6


# 5' flank 1-5, mature 6-13, loop 14-25, star 26-33, 3' flank 34-40. With
# miRDeep2's allowance (start up to 2 nt early, end up to 5 nt late) the
# windows are mature 4-18, star 24-38 and loop 12-30.
EXP = "fffff" + "M" * 8 + "l" * 12 + "S" * 8 + "fffffff"


def read(beg, end, count):
    """A read covering precursor positions beg..end (1-based), as output.mrd draws it."""
    return "." * (beg - 1) + "a" * (end - beg + 1) + "." * (len(EXP) - end), count


def test_arm_counts_allow_2_nt_before_and_5_nt_after():
    locus = Locus(EXP, [read(6, 13, 10), read(4, 18, 1), read(3, 13, 100), read(6, 19, 100),
                        read(26, 33, 5), read(24, 38, 2), read(23, 33, 100), read(26, 39, 100),
                        read(16, 23, 3), read(12, 30, 4), read(11, 23, 100), read(16, 31, 100)])
    assert arm_counts(locus) == {"mature": 11, "star": 7, "loop": 7}


def test_read_spanning_two_arms_counts_nowhere():
    locus = Locus(EXP, [read(6, 33, 1000), read(6, 25, 100), read(14, 33, 10)])
    assert arm_counts(locus) == {"mature": 0, "star": 0, "loop": 0}


def test_mature_then_star_then_loop():
    # 14-18 fits the mature and loop windows, 24-30 the star and loop windows
    locus = Locus(EXP, [read(14, 18, 7), read(24, 30, 20)])
    assert arm_counts(locus) == {"mature": 7, "star": 20, "loop": 0}


def test_star_on_5p_arm():
    exp = "fffff" + "S" * 8 + "l" * 12 + "M" * 8 + "fffffff"
    locus = Locus(exp, [read(6, 13, 3), read(26, 33, 9), read(16, 23, 1)])
    assert arm_counts(locus) == {"mature": 9, "star": 3, "loop": 1}


def test_failed_structure_has_no_arms():
    # miRDeep2 writes an exp string of 'n' when it cannot find the hairpin's parts
    locus = Locus("n" * len(EXP), [read(6, 13, 10)])
    assert arm_counts(locus) == {"mature": 0, "star": 0, "loop": 0}


def test_missing_exp_raises():
    with pytest.raises(ValueError):
        arm_counts(Locus(None, [read(6, 13, 10)]))
    with pytest.raises(ValueError):
        five_prime_features(Locus(None, [read(6, 13, 10)]))


def test_homogeneity_is_simpson_index_of_mature_arm_starts():
    # Mature arm: starts 6 (8 reads, two different 3' ends) and 7 (2 reads). The
    # read starting at 3 is outside miRDeep2's mature window; star and loop reads
    # do not count. Unbiased Simpson index: sum n(n - 1) / (N(N - 1)).
    locus = Locus(EXP, [read(6, 13, 6), read(6, 12, 2), read(7, 13, 2), read(3, 13, 100),
                        read(26, 33, 50), read(16, 23, 9)])
    homogeneity, num_locis = five_prime_features(locus)
    assert homogeneity == pytest.approx((8 * 7 + 2 * 1) / (10 * 9))
    assert num_locis == 6


def test_homogeneity_of_one_mature_read_is_blank():
    homogeneity, _ = five_prime_features(Locus(EXP, [read(6, 13, 1), read(26, 33, 50)]))
    assert math.isnan(homogeneity)
    homogeneity, _ = five_prime_features(Locus(EXP, [read(6, 13, 2)]))
    assert homogeneity == 1
