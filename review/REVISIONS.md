# AutoDeep code revisions

From the mock BioTechniques review of "AutoDeep: Automatic Stratification of miRDeep2 Outputs" (16 Oct 2026). Line numbers refer to commit 133bf14 (22 Oct 2025); if lines have moved, search for the quoted code.

Status: `[ ]` to do, `[~]` in progress, `[x]` merged. Branch names and commits carry the item ID, e.g. `a1-boolean-encoding`.

## Open questions (Tejus answers these first)

Write each answer under its question. Claude Code: if an item depends on an unanswered question, stop and ask.

- **Q1. Seed homology.** All 728 training loci with a miRBase seed match are labelled Confident; none of the 1,907 Candidate or false-positive loci has one. Is seed homology a labelling criterion (then drop it as a feature, or evaluate only loci without it) or a feature (then labels must not use it)? Affects B5, C1, C5.
  Answer:
- **Q2. 5'-homogeneity window.** Denominator = reads on the mature arm as marked by miRDeep2's `exp` string, or reads starting within +/- k nt of the dominant start (k = ?). Affects B2.
  Answer (Tejus, 18 Oct 2026): the mature arm as miRDeep2 marks it (the reads `arm_counts` assigns to the mature, which sum to result.csv's `mature_read_count`), summarised by the unbiased Simpson index of their 5' start positions: the probability that two reads drawn from the mature arm without replacement start at the same position, sum n_i(n_i - 1) / (N(N - 1)). Chosen over the dominant read's share and Shannon entropy (raw or normalised) after a comparison on all 2,907 training loci. All of them rank the loci almost identically (Spearman >= 0.99), but only the Simpson estimator has no depth bias: when reads are thinned at random to 10%, it shifts by +0.005 SD, against +0.05 SD for the dominant share and -0.11 SD for Shannon entropy. Normalised entropy (entropy / log2 of the number of distinct starts) also falls when stray reads are added.
- **Q3. Overhangs.** Signed 3' overhang at the Drosha end and at the Dicer end? Sign convention when the overhang is on the 5' side? Affects B3.
  Answer (Tejus, 18 Oct 2026): signed, one value per duplex end (Drosha end and Dicer end): positive is a 3' overhang (+2 canonical), 0 blunt, negative a 5' overhang. Use the observed star and the precursor structure miRDeep2 writes in output.mrd (no refold; miRDeep2's ViennaRNA is up to date). Measure each end from the terminal base pair between the mature and the star (the duplex rule), not from miRDeep2's expected star: pairs with the flanks or loop do not count, since Drosha and Dicer cut those off. Blank (missing) when a locus has no star reads, because miRDeep2's star is then its own prediction, which assumes 2-nt overhangs.
- **Q4. Terminal-U flag.** 5'U only, or separate 5'U and 3'U features? The text says U/A at the ends of precursor, star and mature; the code checks U at either end of the mature only. Affects B4.
  Answer:
- **Q5. miRDeep2 PPV columns.** `mirDeep_sensitivity` and `sensitivity_uncertainty` are miRDeep2's "estimated probability that the miRNA candidate is a true positive" and its SD, not sensitivity. They identify the species with 99% accuracy. Keep (renamed) or drop? Affects B5, C5.
  Answer:
- **Q6. Count normalisation.** Reads per million mapped (needs mapped-read totals per library) or rank within run? Affects B7.
  Answer:
- **Q7. rfam alert.** Output flag only, or force the label to false positive? Affects B6.
  Answer:
- **Q8. Abstention.** Add a "review" label when the top class probability is below tau? Value of tau? Affects C3.
  Answer:
- **Q9. Conflicting labels.** Six groups of identical feature vectors carry different labels (listed in review/conflicts.csv after B8). Resolve by hand. Affects B8.
  Answer:

## Baseline

`python review/reanalysis.py AutoDeep/training_data_pre AutoDeep/model_weights/miRNA_model.json` reproduces `review/reanalysis_baseline.txt`; expect small differences across library versions. Headline numbers (untuned XGBoost: 600 trees, depth 5, eta 0.03, balanced weights):

| Model | Accuracy, random 80/20 x10 | Accuracy, leave-one-species-out | LOSO Confident recall, no seed match |
| --- | --- | --- | --- |
| Depth-3 tree on seed flag + miRDeep2 score | 0.600 | 0.599 | 0.35 |
| Logistic regression, 18 features | 0.791 | 0.754 | 0.36 |
| XGBoost, 9 miRDeep2 columns | 0.745 | 0.664 | 0.43 |
| XGBoost, all 18 features | 0.864 | 0.813 | 0.40 |

## Phase A: code fixes (normal predictions unchanged; items are independent and can run in parallel)

- [x] **A1. Boolean encoding can silently flip.** `scripts/boosted_forest_inference.py` L22-24 use `astype('category').cat.codes`, so codes depend on which values appear in the user's file. In a run where every locus has randfold "yes", "yes" becomes 0, which the model reads as "no". Same pattern in `boosted_forest_training.py` L118-120 and `alt_model_test.py` L129-132.
  Fix: one shared encoder (e.g. `AutoDeep/features.py`) with explicit maps `{"yes": 1, "no": 0}` and `{True: 1, False: 0, "True": 1, "False": 0}`; raise on unknown values; use it in training, inference and evaluation.
  Test: an all-"yes" randfold column encodes to 1; an unknown value raises.
- [x] **A2. Training mutates the installed package.** `boosted_forest_training.py` L91 writes user labels into `training_data/`; L267 overwrites `model_weights/miRNA_model.json` unless `-nw` is given.
  Fix: write user data, model, config and metrics to an output directory (default `./AutoDeepRun/models/<timestamp>/`); never write inside the package. Add `--model PATH` to inference (default: bundled model).
  Test: after `AutoDeep train` in a temp dir, package files are byte-identical.
- [x] **A3. `eval()` on the hyperparameter file.** `boosted_forest_training.py` L154 and `alt_model_test.py` L169.
  Fix: JSON or YAML, validated against the allowed keys (already listed at L147).
  Test: a file containing `__import__('os')` is rejected.
- [x] **A4. `alt_model_test.py` cannot run end to end.** In the tuning branch, L413-481 use an undefined `config` (UnboundLocalError at L422) and assume SVM parameters for every model; L495 saves the model-type string, not the fitted pipeline; the `xgboost` branch is unreachable (click choices, L21).
  Fix (Tejus, 16 Oct 2026: fix in place rather than retire): tune and save the requested model; write only to a run folder; train on `training_data_pre/` through the same loader as `AutoDeep train`; seed as in A6. The models move into the evaluation harness at C1.
  Test: tuning runs end to end for svm, knn and mlp and saves a fitted pipeline of that type; the package is unchanged; the bundled data plus the Pomonella labels give the same 2,907 rows as `AutoDeep train`.
- [x] **A5. `AutoDeep visualize` loop.** `boosted_forest_visualization.py` L46-54 iterates `num_boosted_rounds()` over a dump holding 3 x rounds trees, and re-dumps the model on every iteration.
  Fix: dump once; label each tree by round and class (`tree_info`); make tree export opt-in. SHAP plots (C4) become the default output.
- [x] **A6. Determinism.** No seeds in `train_test_split`, XGBoost or Optuna.
  Fix: a `--seed` option threaded through splits, `random_state` and `OptunaSearch(sampler=optuna.samplers.TPESampler(seed=...))`.
  Test: two runs with the same seed give identical metrics.
- [x] **A7. Packaging and tests.** `setup.py` has a placeholder author and URL and an MIT classifier, but LICENSE is GPL-3.0; `requirements.txt` lists only pandas; there are no tests or CI.
  Fix: correct metadata, full requirements, `pytest` with GitHub Actions, and a small example (miRDeep2's tutorial output) with expected AutoDeep output.
  Note: `setup.py` was already deleted in 36d9492 (1 Mar 2026), after the reviewed commit. AutoDeep is installed with conda from `AutoDeep/AutoDeep.yml`, which lists every dependency, and run from PATH; `requirements.txt` (pandas only) is removed.
- [x] **A8. Locus IDs with commas (added 18 Oct 2026; do before B8).** `init_data.py` removes every comma from `result_*.csv` (`content.replace(",", "")`), but `output.mrd` keeps miRDeep2's IDs. Since B1, `folding_properties.py` therefore finds none of a run's loci when the contig names contain commas, as NCBI RefSeq names often do, and `AutoDeep infer` stops: on the S. frugiperda run, "145 loci from result.csv are not in output.mrd", e.g. `NC_049717.1_Spodoptera_frugiperda_isolate_Faw-zju_chromosome_8,_ZJU_Sfru_1.0,_whole_genome_shotgun_sequence_12098`. The pypdf code worked because miRDeep2 leaves the commas out of the PDF file names. The FAW rows of `training_data_pre/` were made from a copy of `output.mrd` with every comma removed. B8 rebuilds FAW from the original run, so it needs this fix.
  Fix: normalise the `output.mrd` IDs as `init_data.py` normalises `result.csv` (one shared function: remove commas), and raise if two IDs become equal. Output IDs stay without commas, as before B1. The in-place rewrite of `result_*.csv` is a separate known issue (CLAUDE.md).
  Test: a copy of the tutorial run with commas added to one locus ID in `result_*.csv` and `output.mrd` gives the same `xgboost_judgement.csv` as the original, apart from that ID; two `output.mrd` IDs that differ only in commas raise an error. Local check: `AutoDeep infer` runs on the FAW run and reproduces the 5' homogeneity of all 145 rows of `original_data_FAW.csv`.

## Phase B: feature definitions (predictions change; needs the raw miRDeep2 runs in CLAUDE.local.md)

Do B1 first and B8 last.

- [x] **B1. Parse `output.mrd`, not PDFs.** `folding_properties.py` L187-240 extract read alignments from `pdfs_*/<locus>.pdf` with pypdf, which breaks across miRDeep2 and PDF-library versions.
  Fix: a parser for miRDeep2's `output.mrd` (per-locus `exp` string marking flank, mature, loop and star; read lines with counts).
  Test: on miRDeep2's tutorial run, per-arm read counts from the parser equal the mature, star and loop counts in `result.csv`.
- [x] **B2. 5'-homogeneity denominator (Q2).** `folding_properties.py` L231, `tdf = selection_df[(selection_df['Position'] - max_position) <= 20]`, keeps every read that starts upstream of the dominant read. For 3p-arm miRNAs, star and loop reads therefore enter the denominator and bias the feature low. Also, L230's `top_loci` (sized by the PDF page count) feeds a feature that is never saved (L244), although the manuscript describes it.
  Test: adding star-arm reads to a synthetic 3p-dominant locus leaves homogeneity unchanged.
  Decision (Tejus, 18 Oct 2026; Q2): the unbiased Simpson index of the 5' start positions of the mature-arm reads, grouped by position whatever the first nucleotide; blank with fewer than 2 mature-arm reads. `num_locis` is unchanged. The unsaved `top_loci` went with the pypdf code in B1. The comparison of measures is `review/homogeneity_measures.py`; the reasons are in `review/METHODS_CHANGES.md` (B2).
- [x] **B3. Overhang features (Q3).** L41-43 compute "Bot" as `abs(right - left)`, the difference between the unpaired 5' and 3' ends of the whole fold; L69-154 compute "Top" with a padding heuristic. Seven S. frugiperda training rows have negative "Bot" values, so those training features came from different code.
  Fix: signed 3' overhangs at each duplex end from mature and star coordinates and the base-pair table, as miRDeep2 does (`miRDeep2_core_algorithm.pl`, `star_beg_exp` and `star_end_exp`).
  Decision (Tejus, 18 Oct 2026): the duplex rule instead of miRDeep2's expected-star rule. At the Drosha end take the outermost base pair between the mature and the star, at the Dicer end the innermost, and count the unpaired nucleotides beyond it on each strand (3'-end strand minus 5'-end strand). The two rules differ on 10 of 148 Pomonella loci with star reads; reasons in `review/METHODS_CHANGES.md` (B3).
  Test: ten MirGeneDB 3.0 human precursors chosen in advance give the overhangs their folds show: 2 at both ends for seven; Let-7-P2a1 (let-7a-1) and Mir-145 give 1 at the Drosha end, and Mir-21 gives 3 at the Dicer end (Tejus, 18 Oct 2026).
- [ ] **B4. Terminal-U flag (Q4).** `csv_feature_extraction.py` L51: `x[0] == 'u' or x[-1] == 'u'`.
- [ ] **B5. Feature names (Q1, Q5).** One name per feature across code, Table 1 and figures (map below). Keep a shim that reads old CSVs.
- [ ] **B6. rfam alert (Q7).** `csv_feature_extraction.py` L23 drops `rfam_alert`, which miRDeep2's documentation says should overrule its true-positive estimate. Pass it through to `xgboost_judgement.csv`.
- [ ] **B7. Depth dependence (Q6).** Raw counts scale with library size: D. melanogaster MirGeneDB loci have a median of 116,992 total reads versus 341 in the curated set. Implement Q6. Add a down-sampling check: binomially thin `output.mrd` read counts to 50%, 25% and 10%, recompute features and report how many labels change. The full version subsamples reads and reruns miRDeep2.
- [ ] **B8. Regenerate the training data.** Rebuild all 11 CSVs from the raw runs with the released code. Add `species`, `dataset` and `dup_group` columns (identical feature vectors share a group). Write `review/conflicts.csv` for Q9; do not relabel anything. Human runs include ALT-contig copies (e.g. chr16_7743 and chr16_KI270853v1_alt_21843): flag them now, and remap to the primary assembly without ALT contigs at the next miRDeep2 run.
  Before B8, decide whether to add these 5' features (Tejus, 18 Oct 2026, from the Q2 discussion; each one changes the feature set, Table 1 and B5's name map):
  - **Share of mature-arm reads within +/-1 nt of the commonest 5' start.** It tolerates single-nucleotide 5' isomiRs, so next to the Simpson index it separates "precise, with a 5' isomiR" from "degraded". It is the only candidate that ranks the 2,907 training loci differently from the Simpson index (Spearman 0.69).
  - **The Simpson index on the star arm**, blank without star reads, as for the B3 overhangs. It is independent evidence from the other cleavage. The entropy code kept in `folding_properties.py` has a star version (`normalized_entropy_s`).

### Feature name map (proposed; confirm in B5)

| Current column | Table 1 name | Proposed column |
| --- | --- | --- |
| five_prime_process_signal_selected_row_count_sum_over_mature_count_sum | Five Prime Homogeneity | five_prime_homogeneity |
| num_locis | Unique Alignments | n_distinct_reads |
| mirDeep_sensitivity, sensitivity_uncertainty | miRDeep2 Sensitivity, Sensitivity sd | mirdeep2_est_ppv, mirdeep2_est_ppv_sd (or drop, Q5) |
| mature_5'u_or_3'u | Mature 5'/3' U? | mature_5p_u (plus mature_3p_u, Q4) |
| 3' Overhang Bot, 3' Overhang Top | 3' Overhang Bot, Top | overhang_3p_drosha_end, overhang_3p_dicer_end (Q3) |
| Folding Energy | Folding Energy | mfe_kcal_mol |

## Phase C: evaluation (produces the paper's tables and figures)

Build `AutoDeep/scripts/evaluate.py` starting from `review/reanalysis.py`.

- [ ] **C1. Nested leave-one-species-out.** Outer loop holds out one species (and, as a second analysis, one dataset); the inner loop tunes on the rest. Keep each `dup_group` on one side of every split. Give every model the same tuning budget, and log1p-transform counts for non-tree models. Report per-class precision, recall and F1, macro-F1, balanced accuracy and bootstrap 95% CIs, all split by seed homology. Lock the Phase 2 datasets and any new taxon as a final test set, untouched until the end.
- [ ] **C2. Baselines and competitors.** miRDeep2 score cut-off (chosen in the inner loop), seed rule, logistic regression, random forest and XGBoost on identical folds. For mirdeepsquared, run it on the same datasets' `result.csv` and `output.mrd`, then merge predictions by `provisional_id`.
- [ ] **C3. Calibration and workload (Q8).** Calibrate on inner folds (isotonic or temperature scaling). Add a reliability diagram, Brier score and a coverage-recall curve: the fraction of loci a curator can skip while keeping 95% of Confident loci.
- [~] **C4. SHAP.** Make SHAP plots the default output of `AutoDeep visualize` (moved here from A5). Per-class summary plots and dependence plots for 5' homogeneity, overhangs and MFE, from XGBoost's built-in `pred_contribs=True` (no new dependency). Add the top three contributions per locus to the output CSV. These replace Figs 2-4; export vector files (PDF or SVG, Helvetica 8 pt).
- [ ] **C5. Ablation.** All features; minus seed homology; minus PPV columns; minus raw counts; miRDeep2 columns only.

## Phase D: final model and release

- [ ] **D1. Final fit and search space.** `boosted_forest_training.py` L125 and L260-261 refit the chosen config on the first, unstratified 80% split, without the balanced sample weights used during tuning. Refit on all training data with the same weights. In the search space (L210-224), `tune.randint(1, 6)` gives max_depth 1-5, `base_score` has no effect under softmax, and ASHA is inert with one report per trial; prefer early stopping to tuning n_estimators. Save config, seed and metrics beside the model.
- [ ] **D2. Release.** Tag v1.0.0, archive on Zenodo for a DOI, and add a CHANGELOG mapping review items to commits. The paper must state the three-year availability commitment BioTechniques requires.

## Not code (Tejus)

- Labelling protocol, and Cohen's kappa on at least 200 double-labelled loci.
- Data for at least one taxon outside the training set (e.g. zebrafish, C. elegans or a second plant).
- A timed curation study, or dropping the time-saving claim.
- Manuscript text: collect required changes in `review/METHODS_CHANGES.md` as items land.
