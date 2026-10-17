# Methods text changes

Sentences in the manuscript's Methods that must change because the code changed. One entry per change: item ID, old text, new text.

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
