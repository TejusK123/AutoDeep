"""B1: 5' homogeneity, num_locis and per-arm read counts from miRDeep2's output.mrd."""
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


def test_homogeneity_and_num_locis(loci):
    homogeneity, num_locis = five_prime_features(loci["locus_1"].reads)
    # Dominant read starts 'a' at 4: reads 1 and 2 (15). Denominator: every read
    # starting at most 20 nt after it, so all but read 6 (10+5+3+2+4 = 24).
    assert homogeneity == pytest.approx(15 / 24)
    assert num_locis == 6


def test_tie_goes_to_first_read_in_file(loci):
    # Reads 7 and 8 both have 6 reads; read 7 comes first, and read 9 starts
    # at the same place but with a different (mismatched) nucleotide.
    homogeneity, num_locis = five_prime_features(loci["locus_2"].reads)
    assert homogeneity == pytest.approx(6 / 13)
    assert num_locis == 3


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
