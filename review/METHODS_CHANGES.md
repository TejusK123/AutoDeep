# Methods text changes

Sentences in the manuscript's Methods that must change because the code changed. One entry per change: item ID, old text, new text.

## B3. Signed 3' overhangs at both ends of the duplex

- **Old text:** TODO (Tejus): paste the sentences that define "3' Overhang Bot" and "3' Overhang Top" (Methods and Table 1).
- **New text:** "For each candidate, AutoDeep computes the 3' overhang at both ends of the mature-star duplex, at the base of the stem (Drosha end) and at the loop (Dicer end), from miRDeep2's output.mrd: the mature, the most abundant star read and the RNAfold structure of the precursor that miRDeep2 used. Only base pairs between the mature and the star are considered, since Drosha and Dicer remove the flanks and the loop. At each end, the overhang is the number of unpaired nucleotides beyond the terminal mature-star base pair on the strand whose 3' end lies there, minus the number on the strand whose 5' end lies there. Positive values are 3' overhangs (2 for canonical processing), 0 is a blunt end and negative values are 5' overhangs. Candidates without star reads have no overhang values, because miRDeep2's star is then its own prediction."
- **Note:** before B3, "Bot" was the unsigned difference between the unpaired 5' and 3' ends of AutoDeep's own fold of the precursor, and "Top" came from a padding heuristic; both were computed even without star reads. The training data in `training_data_pre/` keep those values until B8 rebuilds them, and the shipped model was trained on them, so it has to be retrained (D1) before the new definition is used for the paper's results.

### For the paper (why the duplex rule; Tejus chose it on 18 Oct 2026)

- **Definition in one sentence:** "the 3' overhang at each end of the mature-star duplex, from the terminal base pair between the two strands". The alternative considered, miRDeep2's rule, needs "2 plus the offset of the observed star from the star miRDeep2 expects for 2-nt overhangs". That reads as circular even though it is not, and a reviewer may read it that way.
- **Biology:** an overhang is a property of the cut duplex. Drosha and Dicer each leave a 2-nt 3' overhang on the mature-star duplex, and once the flanks and loop are cut off, pairs with them no longer exist. miRDeep2's rule anchors on fixed mature positions (the first and third-from-last nucleotides), which can pair with the loop; that is the cause in 5 of the 10 Pomonella loci where the rules differ. It also changes when the mature call flips between arms: miRDeep2 calls the mature as the most abundant read, and MirGeneDB lists both arms as mature for 45 human genes. The duplex rule gives the same value whichever strand is the mature.
- **Structure uncertainty:** both rules depend on the predicted structure where it is least reliable: the duplex ends, next to bulges, and where the stem meets the loop or flanks. The choice of rule changes 10 of 148 Pomonella values with star reads (about 7%), about the same as swapping miRDeep2's older RNAfold structure for ViennaRNA 2.7.0's (about 3-4% of the same values). Do not interpret a single locus's value. The duplex rule has fewer extreme values (MirGeneDB 3.0 human, 506 precursors whose fold pairs the two arms: 3 versus 10 beyond +/-5) and gives 2 at both ends slightly more often (378 versus 373).
- **Model:** the main gain of B3 is the sign: before, a 2-nt 5' overhang looked the same as the canonical 2-nt 3' overhang. The overhangs are minor features (about 5% of the shipped model's mean |SHAP|: Top 0.028, Bot 0.026), so the choice of rule should not change accuracy measurably. The duplex rule is less redundant with `miRDeep2_score`, whose star-read component uses miRDeep2's expected star. Report the C5 ablation: if removing both overhangs costs nothing, say so.
- **Caveat to state:** a frayed end, with unpaired nucleotides on both strands beyond the terminal pair, nets to 0, the same as a blunt end (e.g. QFTL02000009.1_39298: 5 and 5).
- **Loci where the rules differ** (C. pomonella; positions 1-based along the obs, pri_seq and pri_struct lines of output.mrd; "old" is the pre-B3 code on AutoDeep's own fold, unsigned; Drosha / Dicer):

  | Locus | Old | miRDeep2's rule | Duplex rule |
  | --- | --- | --- | --- |
  | QFTL02000135.1_4908 | 2 / 1 | 2 / -1 | 2 / 3 |
  | QFTL02000009.1_39390 | 2 / 2 | 2 / -2 | 2 / 2 |
  | QFTL02000001.1_671 | 6 / 5 | 6 / 5 | 6 / 6 |
  | QFTL02000004.1_17907 | 2 / 2 | 2 / 4 | 2 / 2 |
  | QFTL02000009.1_39298 | 3 / 0 | 4 / 5 | 4 / 0 |
  | QFTL02000009.1_39327 | 2 / 2 | 2 / 2 | 1 / 2 |
  | QFTL02000015.1_6015 | 2 / 2 | 2 / 2 | 2 / 1 |
  | QFTL02000447.1_20491 | 2 / 2 | 2 / 2 | 2 / 1 |
  | QFTL02000007.1_25533 | 2 / 2 | 2 / 2 | -3 / 2 |
  | QFTL02000020.1_10445 | 3 / 1 | 5 / 0 | 5 / -1 |

  In the first five, miRDeep2's anchor pairs with a loop nucleotide; in the last five it is unpaired, and miRDeep2's rule counts the unpaired nucleotides as paired.

## B1. Read alignments from output.mrd

- **Old text:** Using mirDeep2's alignment of reads to predicted precursor, 5' homogeneity was quantified.
- **New text:** "For each candidate locus, AutoDeep reads the aligned reads and their counts from miRDeep2's output.mrd file, which lists every read mapped to the precursor in precursor coordinates."
- **Note:** feature values are unchanged. On the C. pomonella run, 5' homogeneity and the number of distinct reads computed from output.mrd equal the PDF-derived training values for all 212 loci (difference < 1e-9).

## A6. Fixed random seeds

- **Old text:** none
- **New text:** "All random steps (data splits, XGBoost's row and column subsampling, and Optuna's TPE sampler) use a fixed seed (42 by default), which is saved with each model. Every hyperparameter configuration is scored by its mean weighted F1 over the same ten stratified 80/20 splits."
- **TODO (Tejus): put your hardware in the reproducibility statement.** Give the CPU model and core count, the OS, and the software versions (`AutoDeep.yml`). The same seed does not give the same model everywhere:
  - Tuning runs as many trials at once as there are CPUs. Optuna's first 10 proposals are seeded random draws, but later ones are fitted to the trials that have finished so far, so with more than 10 tuning rounds the chosen hyperparameters depend on the number of CPUs. In a test with the same seed, 9 of 24 proposed configurations differed between 3 and 10 CPUs. On the same machine (Apple M4, 10 cores), two 20-round runs with seed 42 tried the same 40 configurations and chose the same hyperparameters. On one machine this is likely but not guaranteed, because it depends on the order in which trials finish.
  - XGBoost's results differ slightly between machines. The reviewer's baseline (`review/reanalysis_baseline.txt`, x86, NumPy 2.x) and this Mac (`review/reanalysis_osx-arm64.txt`, Apple M4, NumPy 1.23.5) differ by up to 0.01 in the XGBoost metrics.
  - Suggested wording, once D1 has retrained the model: "The released model was trained with seed 42 on an Apple M4 (10 cores) running macOS 26.3, with the software versions listed in AutoDeep.yml. Tuning with more than ten trials can select different hyperparameters on machines with a different number of cores." Check the machine and versions against the final training run.
- **Note:** the shipped model was trained before this change, without seeds. D1 retrains it and records its seed.

## A4. Comparison models (SVM, KNN, MLP) train on the XGBoost data

- **Old text:** TODO (Tejus): paste the sentence that describes how the SVM, k-nearest-neighbour and multilayer-perceptron models in Table 2 were trained and tuned, or write "none".
- **New text:** "For comparison, support vector machine, k-nearest-neighbour and multilayer perceptron classifiers (scikit-learn 1.5.1) were trained on the same 2,907 loci and 18 features as the XGBoost model, after mean imputation and standardisation, and tuned in the same way (Ray Tune with Optuna's TPE sampler, scored by mean weighted F1 over the same ten stratified 80/20 splits, seed 42)."
- **Note:** Table 2 must be regenerated with the fixed script. Before A4 the script trained on `training_data/` (3,180 loci, with the entropy 5' features in place of 5' homogeneity and the number of distinct reads), its tuning crashed before saving, and it never saved a usable model. Check which data and code produced the current Table 2.
- **Note:** class weighting still differs between models: XGBoost (sample weights) and SVM (`class_weight='balanced'`) weight the classes equally; KNN and MLP do not. Equal treatment of all models is part of C1.
