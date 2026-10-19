# AutoDeep

XGBoost post-filter for miRDeep2. It labels each novel pre-miRNA candidate Candidate, Confident or falsepositive, with class probabilities, from 18 read-, structure- and sequence-derived features. The paper (BioTechniques Report) is under revision, and the code must match its Methods text exactly.

## Current work

We are revising the code in response to a mock peer review. The work list, with file/line references, open questions and acceptance tests, is in `review/REVISIONS.md`. Read it before starting any task. Put the item ID (A1, B2, ...) in branch names and commit messages. REVISIONS.md line numbers refer to commit 133bf14; the code has moved since, so search for the quoted code.

### Status (19 Oct 2026, after C4)

- Phase A is complete; all merged into `refactoring`:
  - A1: fixed flag/label encoding
  - A2: training writes to a run folder
  - A3: no `eval()` on hyperparameter files
  - A4: `alt_model_test.py` fixed in place, not retired (Tejus's decision); trains SVM/KNN/MLP on the same data as `AutoDeep train`
  - A5: visualize trees
  - A6: `--seed`, default 42
  - A7: CI on Linux and macOS, `AutoDeep.yml` as the one requirements list, miRDeep2 tutorial example
  - A8 (added 18 Oct 2026, PR #12): locus IDs with commas. output.mrd IDs lose their commas as result.csv's do (`mrd.strip_commas`), so `AutoDeep infer` runs on FAW (it had failed since B1); FAW's features equal its training rows except the B3 overhangs
- B1 is complete (PRs #2 and #10): the 5' features come from `output.mrd` and equal the pypdf values, and `arm_counts` reproduces miRDeep2's mature/star/loop counts in `result.csv` (tutorial: 8 loci; Pomonella: 212).
- B2 is complete (PR #13): Q2 answered. 5' homogeneity is the unbiased Simpson index of the 5' start positions of the mature-arm reads (miRDeep2's arm rule, `mrd.reads_by_arm`), grouped by position whatever the first base (Tejus's choice), blank with fewer than 2 mature-arm reads; `num_locis` and the column name are unchanged (B5 renames). The reasons and the comparison of measures are in METHODS_CHANGES.md (B2) and `review/homogeneity_measures.py`. The training CSVs keep the old values until B8; until D1 retrains, the shipped model changes 88/2,907 training labels (Pomonella 8/212). Two optional extra 5' features, to decide before B8, are listed under B8 in REVISIONS.md.
- B3 is complete (PR #11): Q3 answered. Signed 3' overhangs at the Drosha end ("Bot") and the Dicer end ("Top") by the duplex rule (Tejus's choice over miRDeep2's `find_star` rule: count from the terminal mature-star base pair, ignoring pairs with the flanks or loop), from the `obs` star and `pri_struct` in `output.mrd` (no refold: Tejus's choice, miRDeep2's ViennaRNA is current); blank without star reads. The reasons are in METHODS_CHANGES.md (B3, "For the paper"). The training CSVs keep the old unsigned values until B8. Until D1 retrains, the shipped model sees blanks it was never trained on: on Pomonella 29/212 labels change (28 Candidate and 1 Confident to falsepositive), all of them loci without star reads.
- C4 is complete (PR #14): `AutoDeep visualize` writes SHAP plots by default (per-class summary plots = the new Figs 2-4, mean absolute SHAP per class, dependence plots for 5' homogeneity, the overhangs and folding energy; vector PDF/SVG, 8 pt Helvetica kept as text) for the bundled training data unless `--data`; gain/weight/cover only with `--legacy_importance` (Tejus's request). `xgboost_judgement.csv` lists each locus's three largest SHAP contributions (Tejus asked whether one header would do: no, 458 different top-three lists across the 2,907 training loci). The paper's figures and the Results paragraph that reads the trees wait for D1. On the shipped model, the seed match dominates the Confident class (mean absolute SHAP 1.78), which bears on Q1.
- Also merged, outside REVISIONS: portable conda env for osx-arm64; training reads `training_data_pre/`.
- Next session: ask Tejus where to start. B4-B8 wait on answers to Q1 and Q4-Q9 (B4 on Q4, the terminal-U flag: options and data were given to Tejus on 18 Oct; Claude recommended the mature's 5' nucleotide as a category). The code changes in D1 need no answers.
- CI (`.github/workflows/tests.yml`) is green: on PR #14 (142 passed, 8 skipped, the tests that need the local testing folder, on ubuntu-latest and macos-latest; 150 passed locally with it) and on `refactoring` after the merge (a1baf75). It runs on every push and PR to `refactoring` and `main`; check it with `gh pr checks <n>` before handing a PR over.

### Waiting on Tejus (not code)

- Regenerate Table 2 (SVM/KNN/MLP vs XGBoost) with the fixed `alt_model_test.py`; the old script trained on `training_data/` and never saved a usable model.
- `review/METHODS_CHANGES.md`: old Methods text for A4 and B3 (overhang definitions, also in Table 1); hardware for the A6 reproducibility statement once D1 has retrained the model.
- Answers to Q1 and Q4-Q9 in `review/REVISIONS.md`.
- Whether `testing/Validation/` (CLAUDE.local.md) holds the Phase 2 test sets, and where the miRDeep2 runs behind them are; C1 needs their output.mrd and result_*.csv.
- MirGeneDB's terms of use for the ten sequences in `tests/data/mirgenedb_hsa_ten.csv` (the site states no licence).
- The `testfeatures` branch on GitHub (kept; not merged): one commit, 25ad41a (1 Mar 2025, "testing_features"), rewrites eight `training_data/original_data_*.csv` files and the model weights, and that content exists in no other branch. Possibly relevant to B8 (where the seven negative FAW "Bot" values came from). Keep, archive as a tag, or delete: Tejus to decide.

### Working notes

- Tuning runs as many trials at once as Ray allows (Tejus's choice: speed). Up to 10 tuning rounds repeat on any machine; beyond 10, results depend on the CPU count. Do not add a concurrency limiter.
- To show a bug in code that writes inside `AutoDeep/` (as the old `alt_model_test.py` did), run it on a copy of the repo in the scratchpad, never in place.
- After a PR merges: confirm with `gh pr view <n>`, check `git log origin/refactoring..<branch>` is empty before `git branch -d`, mark `[x]`, update this file, push.

### Open decisions (ask Tejus; do not decide)

- Q1 and Q4-Q9 in REVISIONS.md are unanswered; B4-B8 and C1/C3/C5 depend on them. Q1 (seed homology: labelling criterion or feature) affects the most items.
- Feature set: since B1 the pipeline produces the shipped model's features (`five_prime_process_signal_...`, `num_locis`). Since B2, 5' homogeneity is the Simpson index of the mature-arm starts (old column name until B5), unlike the training data until B8. Since B3 the two overhang features are defined differently from the training data (signed, from `output.mrd`, blank without star reads) until B8 rebuilds it. The entropy features in `training_data/` are computed by functions kept in `folding_properties.py` but not written out.

### Known issues not yet in REVISIONS.md

- `init_data.py` rewrites the user's miRDeep2 `result_*.csv` in place (strips commas with `mrd.strip_commas`). Tests run on copies.
- If `AutoDeep train` fails partway, the run folder is left behind.
- `AutoDeep infer ... --model missing.json` only fails at the last step, after the whole pipeline has run.
- The README's test badge shows the `refactoring` branch; point it at `main` at release (D2).

## Layout

- `AutoDeep/AutoDeep`: bash wrapper; put `AutoDeep/` on PATH to get the `AutoDeep` command (there is no setup.py)
- `AutoDeep/scripts/AutoDeep.sh`: command dispatcher. `infer` runs init_data.py, RNAfold, csv_feature_extraction.py, folding_properties.py, boosted_forest_inference.py
- `AutoDeep/scripts/features.py`: fixed encodings of the flag columns and labels (Candidate 0, Confident 1, falsepositive 2), shared by all scripts (A1); `TABLE1_NAMES`, each feature's name in the paper's Table 1, for figures (C4)
- `AutoDeep/scripts/mrd.py`: parses miRDeep2's `output.mrd` (per locus: `exp` and `obs` strings, `pri_seq`, `pri_struct` and reads; IDs without commas, as init_data.py leaves `result_*.csv`: `strip_commas`, A8) and computes 5' homogeneity (unbiased Simpson index of the mature-arm reads' 5' starts; B2), `num_locis`, per-arm read counts by miRDeep2's own rule (`arm_counts`, from `reads_by_arm`; B1) and the signed 3' overhangs at both duplex ends from the terminal mature-star base pairs (`overhangs`, the duplex rule; B3)
- `AutoDeep/scripts/hyperparameters.py`: reads `-hp` files (text "name value", JSON, or a run's config.json) without running code (A3)
- `AutoDeep/scripts/boosted_forest_inference.py`: selects the model's features by name; `--model PATH` (default: bundled model). `xgboost_judgement.csv` also lists each locus's three largest SHAP contributions to its predicted label (`shap_top1_feature`, `shap_top1_value`, ...; C4)
- `AutoDeep/scripts/explain.py`: SHAP values from XGBoost's built-in TreeSHAP (`pred_contribs`), shared by inference and visualize (C4)
- `AutoDeep/scripts/boosted_forest_training.py`: `AutoDeep train` (Ray Tune + OptunaSearch). Writes only to `--out_dir` (default `AutoDeepRun/models/<timestamp>/`): miRNA_model.json, config.json, metrics.json, user_data.csv, tuning log. `-t` takes several files; `--seed` (default 42) seeds splits, XGBoost and the Optuna sampler. `load_training_set()` and `tuning_search_alg()` are shared with alt_model_test.py
- `AutoDeep/scripts/boosted_forest_visualization.py`: `AutoDeep visualize` (C4): SHAP summary plots per class, mean absolute SHAP per class, dependence plots for 5' homogeneity, the overhangs and folding energy, plus shap_values.csv and shap_importance.csv, in `plots<timestamp>/`; vector PDF (or `--format svg`), 8 pt Helvetica kept as text; colours from the dataviz skill's reference palette. Explains the bundled training data unless `--data CSV...`. Gain/weight/cover plots only with `--legacy_importance`; tree plots only with `--trees N|all`; `--model PATH`
- `AutoDeep/scripts/alt_model_test.py`: SVM/KNN/MLP comparison for Table 2 (A4). Run as `python alt_model_test.py -m svm|knn|mlp` from an AutoDeepRun folder; same data, seed and run-folder outputs as `AutoDeep train` (`<model>_model.pkl` via joblib). Folds into evaluate.py at C1
- `AutoDeep/training_data_pre/`: 2,907 labelled loci in 11 CSVs from 8 species; the shipped model's training data, read by `AutoDeep train`
- `AutoDeep/training_data/`: 3,180 loci in 13 `new_data_*` CSVs with entropy-based 5' features (`normalized_entropy`, `normalized_entropy_s`); not used by the pipeline or shipped model
- `AutoDeep/model_weights/miRNA_model.json`: shipped model (XGBoost 2.1.1, 983 rounds x 3 classes, depth 5)
- `AutoDeep/AutoDeep.yml`: portable conda spec and the only requirements list (no requirements.txt); `AutoDeep/AutoDeep.linux-64.lock.yml`: exact Linux env the shipped model was trained in
- `review/`: REVISIONS.md, reanalysis.py (reviewer's evaluation script), reanalysis_baseline.txt (its output on commit 133bf14), reanalysis_osx-arm64.txt (its output on Tejus's Mac), METHODS_CHANGES.md, homogeneity_measures.py (B2's comparison of 5' homogeneity measures on the testing folder; output in homogeneity_measures.txt)
- `tests/`: pytest suite (150 tests); `conftest.py` puts `AutoDeep/scripts` on the import path. `tests/data/mirgenedb_hsa_ten.csv`: ten MirGeneDB 3.0 human precursors with ViennaRNA 2.7.0 folds, chosen in advance for the B3 overhang test
- `examples/mirdeep2_tutorial/`: miRDeep2's tutorial output (`mirdeep2_output/`) and AutoDeep's expected `xgboost_judgement.csv`; checked by `tests/test_example.py`. Regenerate the expected file when a Phase B item changes a feature on purpose
- `.github/workflows/tests.yml`: CI, pytest on ubuntu-latest and macos-latest in an env built from `AutoDeep.yml`, on pushes and PRs to `refactoring` and `main`

## Environment

- Conda env `AutoDeep` from `AutoDeep/AutoDeep.yml` (works on osx-arm64 and linux-64): Python 3.9, numpy 1.23.5, xgboost 2.1.1, ViennaRNA 2.7.0, ray 2.37, optuna 4.1, pandas 2.2.2, scikit-learn 1.5.1, pdf2image 1.17.0 (exact pin required), pytest. No pypdf.
- On Tejus's Mac the env is not activated in Claude's shell: call `/Users/tkoti/miniconda3/envs/AutoDeep/bin/python` directly, and put that `bin/` on PATH when running `AutoDeep.sh` (RNAfold and graphviz's `dot` live there).
- Tests: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests` from the repo root. Set `AUTODEEP_TEST_RUN` to the local testing folder (path in CLAUDE.local.md) to include the tests that need it (they read its `mirdeep_Pomonella/` and `mirDeep_FAW/`); without it they are skipped (as in CI). That folder holds the miRDeep2 run behind every training CSV (all eight species), the AutoDeep runs that made the CSVs and the probable Phase 2 validation sets; CLAUDE.local.md maps them. Tejus allows using anything in it for any test or check, and wants it used (e.g. check a feature change on all eight species, not only Pomonella); read it in place, never write into it.
- A separate conda env `mirdeep2` (miRDeep2 2.0.1.3 from bioconda) is on Tejus's Mac, used to make the tutorial example; see `examples/mirdeep2_tutorial/README.md`.
- Reanalysis on this Mac differs slightly from `reanalysis_baseline.txt` in the XGBoost rows only (by up to 0.01; run-to-run identical). The baseline ran on x86 with NumPy 2.x, so both architecture and library versions differ. Compare later runs with `review/reanalysis_osx-arm64.txt`.
- Machine-specific paths, such as the miRDeep2 run directories behind each training CSV, are in CLAUDE.local.md

## Git workflow

- GitHub: TejusK123/AutoDeep. Work branches start from `refactoring` and pull requests go into `refactoring` (GitHub's default branch is `main`; do not target it). `gh` is installed and logged in.
- Per item: branch `<id>-<short-name>`, mark the item `[~]` in that branch, open a PR with the PR description written to the scratchpad.
- After Tejus merges: `git pull` on `refactoring`, delete the branch locally and on GitHub if still there, mark the item `[x]`, and commit that one-line change directly to `refactoring`.
- Tejus is new to Git: explain branch state plainly and keep the workflow simple.

## Rules

- One REVISIONS.md item per branch. Propose a plan and wait for my OK before editing files.
- For every bug fix, first write a test that fails because of the bug, then fix it.
- Never hand-edit `AutoDeep/training_data_pre/`, `AutoDeep/training_data/` or `AutoDeep/model_weights/`. Regenerate them only with scripts, and only when the item says so.
- If an item depends on an unanswered question in REVISIONS.md, stop and ask me. Never change label definitions, class names or the feature set on your own; those are scientific decisions.
- Set seeds everywhere: data splits, XGBoost `random_state`, the Optuna sampler.
- After any change to features, training or evaluation, run
  `python review/reanalysis.py AutoDeep/training_data_pre AutoDeep/model_weights/miRNA_model.json`
  and report how the numbers moved relative to `review/reanalysis_baseline.txt`.
- When a change alters behaviour the paper describes, add the Methods sentence that must change to `review/METHODS_CHANGES.md` (old text, new text, item ID).
- Mark items `[~]` when started and `[x]` when merged in REVISIONS.md.
- Keep the GPL-3.0 license. Do not commit raw sequencing data or miRDeep2 run directories. One exception (Tejus, 16 Oct 2026, A7): `examples/mirdeep2_tutorial/mirdeep2_output/` holds `result_*.csv` and `output.mrd` from miRDeep2's public tutorial run.
