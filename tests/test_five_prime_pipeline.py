"""B2: 5' homogeneity counts only the reads on the mature arm.

Before B2 the denominator held every read starting no more than 20 nt after the
most abundant read, so for a miRNA on the 3' arm the star and loop reads, which
start upstream, lowered it. Runs folding_properties.py on two synthetic loci
with the same 3'-arm mature reads, one of them also with star and loop reads.
"""
import pandas as pd
import pytest

from test_overhangs_pipeline import COMPUTED, HAIRPIN, TRAINING_CSV, hairpin_sequence, run, substructure

# Star on the 5' arm, mature on the 3' arm; 1-based positions on HAIRPIN
STAR, MATURE = (13, 34), (51, 72)
LOOP = (STAR[1] + 1, MATURE[0] - 1)
# (start, end, reads): 90 mature reads start at 51 (two 3' ends), 10 start at 52
MATURE_READS = [(51, 72, 60), (51, 71, 30), (52, 72, 10)]
STAR_AND_LOOP_READS = [(13, 34, 20), (14, 34, 5), (38, 45, 4)]  # fewer than the mature, as miRDeep2 calls it
# Probability that two mature-arm reads, drawn without replacement, start at the same position
SIMPSON = (90 * 89 + 10 * 9) / (100 * 99)
HOMOGENEITY = "five_prime_process_signal_selected_row_count_sum_over_mature_count_sum"


def exp_line():
    line = ["f"] * len(HAIRPIN)
    for (beg, end), mark in ((STAR, "S"), (LOOP, "l"), (MATURE, "M")):
        line[beg - 1:end] = mark * (end - beg + 1)
    return "".join(line)


def write_inputs(tmp_path):
    seq = hairpin_sequence()
    run_dir, out_dir = tmp_path / "mirdeep2_run", tmp_path / "AutoDeepRun"
    run_dir.mkdir()
    out_dir.mkdir()
    loci = {"syn_3p": MATURE_READS, "syn_3p_with_star_and_loop": MATURE_READS + STAR_AND_LOOP_READS}
    mrd, rnafold = [], []
    for pid, reads in loci.items():
        mrd += [f">{pid}", f"exp                            {exp_line()}"]
        if reads is not MATURE_READS:
            mrd.append(f"obs                            {exp_line()}")
        mrd += [f"pri_seq                        {seq}", f"pri_struct                     {HAIRPIN}\t#MM"]
        for n, (beg, end, count) in enumerate(reads):
            mrd.append(f"seq_{n}_x{count}                    {'.' * (beg - 1) + seq[beg - 1:end] + '.' * (len(seq) - end)}\t0")
        mrd.append("")
        rnafold += [f">{pid}", seq[STAR[0] - 1:MATURE[1]], f"{substructure(STAR[0], MATURE[1])} (-30.00)"]
    (run_dir / "output.mrd").write_text("\n".join(mrd) + "\n")
    (out_dir / "RNAfold_novel_precursor_miRNAs.txt").write_text("\n".join(rnafold) + "\n")
    # The other features: two real loci's values, which 5' homogeneity does not depend on
    features = pd.read_csv(TRAINING_CSV).iloc[:len(loci), :-1].drop(columns=COMPUTED)
    features["provisional_id"] = list(loci)
    features.to_csv(out_dir / "feature_engineered_miRNA_Deep_data_novel_miRNAs.csv", index=False)
    return run_dir


def test_star_and_loop_reads_leave_homogeneity_unchanged(tmp_path):
    run_dir = write_inputs(tmp_path)
    run("folding_properties.py", str(run_dir), cwd=tmp_path)
    out = pd.read_csv(tmp_path / "AutoDeepRun" / "fully_formatted_data.csv").set_index("provisional_id")
    assert out.loc["syn_3p_with_star_and_loop", HOMOGENEITY] == pytest.approx(out.loc["syn_3p", HOMOGENEITY])
    assert out.loc["syn_3p", HOMOGENEITY] == pytest.approx(SIMPSON)
    # The number of distinct reads still counts every read on the precursor
    assert out.loc["syn_3p_with_star_and_loop", "num_locis"] == len(MATURE_READS + STAR_AND_LOOP_READS)
