"""B1: 5' homogeneity and num_locis from miRDeep2's output.mrd."""
import pytest

from mrd import five_prime_features, parse_mrd

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
    assert [count for _, count in loci["locus_1"]] == [10, 5, 3, 2, 4, 7]
    assert loci["locus_1"][2] == ("....Cgcu..........", 3)


def test_homogeneity_and_num_locis(loci):
    homogeneity, num_locis = five_prime_features(loci["locus_1"])
    # Dominant read starts 'a' at 4: reads 1 and 2 (15). Denominator: every read
    # starting at most 20 nt after it, so all but read 6 (10+5+3+2+4 = 24).
    assert homogeneity == pytest.approx(15 / 24)
    assert num_locis == 6


def test_tie_goes_to_first_read_in_file(loci):
    # Reads 7 and 8 both have 6 reads; read 7 comes first, and read 9 starts
    # at the same place but with a different (mismatched) nucleotide.
    homogeneity, num_locis = five_prime_features(loci["locus_2"])
    assert homogeneity == pytest.approx(6 / 13)
    assert num_locis == 3
