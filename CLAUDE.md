# AutoDeep

XGBoost post-filter for miRDeep2. It labels each novel pre-miRNA candidate Candidate, Confident or falsepositive, with class probabilities, from 18 read-, structure- and sequence-derived features. The paper (BioTechniques Report) is under revision, and the code must match its Methods text exactly.

## Current work

We are revising the code in response to a mock peer review. The work list, with file/line references, open questions and acceptance tests, is in `review/REVISIONS.md`. Read it before starting any task. Put the item ID (A1, B2, ...) in branch names and commit messages.

## Layout

- `AutoDeep/scripts/AutoDeep.sh`: pipeline entry point (init_data.py, then RNAfold, csv_feature_extraction.py, folding_properties.py, boosted_forest_inference.py)
- `AutoDeep/scripts/boosted_forest_training.py`: `AutoDeep train` (Ray Tune + OptunaSearch)
- `AutoDeep/scripts/boosted_forest_visualization.py`: `AutoDeep visualize`
- `AutoDeep/scripts/alt_model_test.py`: SVM/KNN/MLP comparison for Table 2 (to be retired, item A4)
- `AutoDeep/training_data/`: 2,907 labelled loci in 11 CSVs from 8 species
- `AutoDeep/model_weights/miRNA_model.json`: shipped model (XGBoost 2.1.1, 983 rounds x 3 classes, depth 5)
- `review/`: REVISIONS.md, reanalysis.py (reviewer's evaluation script), reanalysis_baseline.txt (its output on commit 133bf14)

## Environment

- Conda env from `AutoDeep/AutoDeep.yml`: Python 3.9, xgboost 2.1.1, ViennaRNA 2.7.0, ray 2.37, optuna 4.1, pandas 2.2.2, scikit-learn 1.5.1
- Development install: `pip install -e AutoDeep`
- Tests: `pytest` from the repo root (create `tests/` if it does not exist)
- Machine-specific paths, such as the miRDeep2 run directories behind each training CSV, are in CLAUDE.local.md

## Rules

- One REVISIONS.md item per branch. Propose a plan and wait for my OK before editing files.
- For every bug fix, first write a test that fails because of the bug, then fix it.
- Never hand-edit `AutoDeep/training_data/` or `AutoDeep/model_weights/`. Regenerate them only with scripts, and only when the item says so.
- If an item depends on an unanswered question in REVISIONS.md, stop and ask me. Never change label definitions, class names or the feature set on your own; those are scientific decisions.
- Set seeds everywhere: data splits, XGBoost `random_state`, the Optuna sampler.
- After any change to features, training or evaluation, run
  `python review/reanalysis.py AutoDeep/training_data AutoDeep/model_weights/miRNA_model.json`
  and report how the numbers moved relative to `review/reanalysis_baseline.txt`.
- When a change alters behaviour the paper describes, add the Methods sentence that must change to `review/METHODS_CHANGES.md` (old text, new text, item ID).
- Mark items `[~]` when started and `[x]` when merged in REVISIONS.md.
- Keep the GPL-3.0 license. Do not commit raw sequencing data or miRDeep2 run directories.
