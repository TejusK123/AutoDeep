# Example: miRDeep2's tutorial run

A small, public miRDeep2 run to try AutoDeep on, and the output AutoDeep is expected to give. `tests/test_example.py` checks it on every CI run.

## Contents

- `mirdeep2_output/`: the two files AutoDeep reads from a miRDeep2 run, `result_16_10_2026_t_19_57_25.csv` and `output.mrd` (from `mirdeep_runs/run_16_10_2026_t_19_57_25/`), unchanged.
- `expected/xgboost_judgement.csv`: `AutoDeep infer` on that folder with the bundled model (`AutoDeep/model_weights/miRNA_model.json`).

The tutorial yields one novel candidate, `chrII:11534525-11540624_7`, which AutoDeep labels Confident (probabilities: Candidate 0.400, Confident 0.579, false positive 0.021). miRDeep2 also detects 6 known C. elegans miRNAs; AutoDeep only classifies novel candidates.

## Run AutoDeep on it

From the repository root, with the AutoDeep environment active and AutoDeep on your PATH. Copy the folder first, because AutoDeep rewrites `result_*.csv` in place:

```sh
cp -r examples/mirdeep2_tutorial/mirdeep2_output /tmp/tutorial_run
AutoDeep infer /tmp/tutorial_run
diff AutoDeepRun/xgboost_judgement.csv examples/mirdeep2_tutorial/expected/xgboost_judgement.csv
```

Probabilities can differ in the last digits between x86 and Apple silicon.

## How the miRDeep2 output was made

- miRDeep2 2.0.1.3 from bioconda (`conda create -n mirdeep2 -c conda-forge -c bioconda mirdeep2=2.0.1.3`), with bowtie 1.3.1, randfold 2.0.1 and ViennaRNA 2.7.2, on macOS (Apple M4), 16 Oct 2026.
- Input: `tutorial_dir/` of https://github.com/rajewsky-lab/mirdeep2 at tag `v0.1.3` (C. elegans: `cel_cluster.fa`, `reads.fa`, `mature_ref_this_species.fa`, `mature_ref_other_species.fa`, `precursors_ref_this_species.fa`). The reads are not included here.
- Commands, as in that folder's `run_tut.sh`:

```sh
bowtie-build cel_cluster.fa cel_cluster
mapper.pl reads.fa -c -j -k TCGTATGCCGTCTTCTGCTTGT -l 18 -m -p cel_cluster -s reads_collapsed.fa -t reads_collapsed_vs_genome.arf -v -n
miRDeep2.pl reads_collapsed.fa cel_cluster.fa reads_collapsed_vs_genome.arf mature_ref_this_species.fa mature_ref_other_species.fa precursors_ref_this_species.fa -t C.elegans
```

miRDeep2's error log shows `RNAfold: invalid option -- n`: miRDeep2 first calls `RNAfold -noPS`, which ViennaRNA 2.7 rejects, then retries with `--noPS`, which works. The run is unaffected.
