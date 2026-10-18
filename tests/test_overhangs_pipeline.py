"""B3: the pipeline writes the signed overhangs from output.mrd, and inference takes blanks.

Runs folding_properties.py and boosted_forest_inference.py on two synthetic loci
with a known structure: one whose star leaves 1-nt 5' overhangs at both ends,
and one without star reads. Before B3, the pipeline wrote unsigned values from
its own fold of the precursor, and a value even without star reads.
"""
import os
import random
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "AutoDeep", "scripts")
TRAINING_CSV = os.path.join(REPO, "AutoDeep", "training_data_pre", "original_data_Pomonella.csv")
COMPUTED = ["five_prime_process_signal_selected_row_count_sum_over_mature_count_sum", "num_locis",
            "Folding Energy", "3' Overhang Bot", "3' Overhang Top"]

# 5' flank 1-10, 5' arm 11-36, loop 37-46, 3' arm 47-72, 3' flank 73-82; i pairs with 83 - i
HAIRPIN = "." * 10 + "(" * 26 + "." * 10 + ")" * 26 + "." * 10
# id: (mature, star, star reads?); 1-based positions on the hairpin
LOCI = {
    "syn_5p_overhangs": ((13, 34), (48, 69), True),  # star 3 nt short of the 2-nt-overhang star (51-72)
    "syn_no_star": ((13, 34), (51, 72), False),  # star as miRDeep2 predicts it
}


def hairpin_sequence():
    rng = random.Random(0)
    arm = "".join(rng.choice("acgu") for _ in range(26))
    flank = lambda: "".join(rng.choice("acgu") for _ in range(10))
    pair = {"a": "u", "c": "g", "g": "c", "u": "a"}
    return flank() + arm + flank() + "".join(pair[nt] for nt in reversed(arm)) + flank()


def marks(mature, star):
    line = ["f"] * len(HAIRPIN)
    line[mature[1]:star[0] - 1] = "l" * (star[0] - 1 - mature[1])
    for (beg, end), mark in ((mature, "M"), (star, "S")):
        line[beg - 1:end] = mark * (end - beg + 1)
    return "".join(line)


def substructure(beg, end):
    """HAIRPIN's base pairs within beg..end, as a stand-in for RNAfold's fold of that stretch."""
    partner, stack = {}, []
    for i, c in enumerate(HAIRPIN, 1):
        if c == "(":
            stack.append(i)
        elif c == ")":
            j = stack.pop()
            partner[i], partner[j] = j, i
    return "".join(HAIRPIN[i - 1] if beg <= partner.get(i, 0) <= end else "." for i in range(beg, end + 1))


def write_inputs(tmp_path):
    seq = hairpin_sequence()
    run_dir, out_dir = tmp_path / "mirdeep2_run", tmp_path / "AutoDeepRun"
    run_dir.mkdir()
    out_dir.mkdir()
    mrd, rnafold, formatted = [], [], []
    for pid, (mature, star, observed) in LOCI.items():
        line = marks(mature, star)
        mrd += [f">{pid}", f"exp                            {line}"]
        if observed:
            mrd.append(f"obs                            {line}")
        mrd += [f"pri_seq                        {seq}", f"pri_struct                     {HAIRPIN}\t#MM"]
        for n, ((beg, end), count) in enumerate([(mature, 100)] + ([(star, 20)] if observed else [])):
            aln = "." * (beg - 1) + seq[beg - 1:end] + "." * (len(seq) - end)
            mrd.append(f"seq_{n}_x{count}                    {aln}\t0")
        mrd.append("")
        precursor = (mature[0], star[1])
        rnafold += [f">{pid}", seq[precursor[0] - 1:precursor[1]], f"{substructure(*precursor)} (-30.00)"]
        formatted.append({"provisional_id": pid, "consensus_mature_sequence": seq[mature[0] - 1:mature[1]],
                          "consensus_star_sequence": seq[star[0] - 1:star[1]],
                          "consensus_precursor_sequeunce": seq[precursor[0] - 1:precursor[1]]})
    (run_dir / "output.mrd").write_text("\n".join(mrd) + "\n")
    (out_dir / "RNAfold_novel_precursor_miRNAs.txt").write_text("\n".join(rnafold) + "\n")
    pd.DataFrame(formatted).to_csv(out_dir / "formatted_novel_miRNA.csv", index=False)
    # The other features: two real loci's values, which the overhangs do not depend on
    features = pd.read_csv(TRAINING_CSV).iloc[:len(LOCI), :-1].drop(columns=COMPUTED)
    features["provisional_id"] = list(LOCI)
    features.to_csv(out_dir / "feature_engineered_miRNA_Deep_data_novel_miRNAs.csv", index=False)
    return run_dir


def run(script, *args, cwd):
    result = subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *args], cwd=cwd,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pipeline_writes_signed_overhangs_and_blanks(tmp_path):
    run_dir = write_inputs(tmp_path)
    run("folding_properties.py", str(run_dir), cwd=tmp_path)
    out = pd.read_csv(tmp_path / "AutoDeepRun" / "fully_formatted_data.csv").set_index("provisional_id")
    overhangs = ["3' Overhang Bot", "3' Overhang Top"]
    assert out.loc["syn_5p_overhangs", overhangs].tolist() == [-1, -1]
    assert out.loc["syn_no_star", overhangs].isna().all()
    # Blank overhangs reach the model as missing values
    run("boosted_forest_inference.py", cwd=tmp_path)
    judged = pd.read_csv(tmp_path / "AutoDeepRun" / "xgboost_judgement.csv")
    assert sorted(judged["provisional_id"]) == sorted(LOCI)
    assert judged.loc[judged["provisional_id"] == "syn_no_star", overhangs].isna().all(axis=None)
