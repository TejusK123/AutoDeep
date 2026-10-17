# AutoDeep

XGBoost post-filter for miRDeep2. It labels each novel pre-miRNA candidate Candidate, Confident or falsepositive, with class probabilities, from 18 read-, structure- and sequence-derived features. The paper (BioTechniques Report) is under revision, and the code must match its Methods text exactly.

## Current work

We are revising the code in response to a mock peer review. The work list, with file/line references, open questions and acceptance tests, is in `review/REVISIONS.md`. Read it before starting any task. Put the item ID (A1, B2, ...) in branch names and commit messages. REVISIONS.md line numbers refer to commit 133bf14; the code has moved since, so search for the quoted code.

### Status (16 Oct 2026)

- Done and merged: A1 (fixed flag/label encoding), A2 (training writes to a run folder), A3 (no `eval()` on hyperparameter files), A5 (visualize trees), A4 (`alt_model_test.py` fixed in place, not retired: tunes and saves SVM/KNN/MLP on the same data as `AutoDeep train`), A6 (seeds: `AutoDeep train --seed`, default 42, saved in config.json), A7 (CI on Linux and macOS, `AutoDeep.yml` as the one requirements list, miRDeep2 tutorial example). Phase A is complete.
- In progress: B1. The PDF-parity part is merged (features from `output.mrd` equal the pypdf values); B1's own acceptance test (per-arm read counts vs `result.csv`) is still to do.
- Also merged, outside REVISIONS: portable conda env for osx-arm64; training reads `training_data_pre/`.
- Next: B1's acceptance test (per-arm read counts vs `result.csv`) on the tutorial example; the rest of Phase B waits on Q1-Q9.
- Table 2 (SVM/KNN/MLP vs XGBoost) must be regenerated with the fixed `alt_model_test.py`; Tejus to fill in the A4 old Methods text in `review/METHODS_CHANGES.md`.
- Tuning runs as many trials at once as Ray allows (Tejus's choice: speed). Up to 10 tuning rounds repeat on any machine; beyond 10, results depend on the CPU count. Do not add a concurrency limiter.

### Open decisions (ask Tejus; do not decide)

- Q1-Q9 in REVISIONS.md are unanswered; B2-B8 and C1/C3/C5 depend on them.
- Feature set: since B1 the pipeline produces the shipped model's features (`five_prime_process_signal_...`, `num_locis`). The entropy features in `training_data/` are computed by functions kept in `folding_properties.py` but not written out.
- `review/METHODS_CHANGES.md` A6: Tejus adds the training hardware (CPU, cores, OS, versions) to the reproducibility statement once D1 has retrained the model.

### Known issues not yet in REVISIONS.md

- `init_data.py` rewrites the user's miRDeep2 `result_*.csv` in place (strips commas). Tests run on copies.
- If `AutoDeep train` fails partway, the run folder is left behind.
- `AutoDeep infer ... --model missing.json` only fails at the last step, after the whole pipeline has run.
- `AutoDeep infer --help` names the output `AutoDeep_results.csv`; the file is `xgboost_judgement.csv`.

## Layout

- `AutoDeep/AutoDeep`: bash wrapper; put `AutoDeep/` on PATH to get the `AutoDeep` command (there is no setup.py)
- `AutoDeep/scripts/AutoDeep.sh`: command dispatcher. `infer` runs init_data.py, RNAfold, csv_feature_extraction.py, folding_properties.py, boosted_forest_inference.py
- `AutoDeep/scripts/features.py`: fixed encodings of the flag columns and labels (Candidate 0, Confident 1, falsepositive 2), shared by all scripts (A1)
- `AutoDeep/scripts/mrd.py`: parses miRDeep2's `output.mrd` and computes 5' homogeneity and `num_locis` (B1)
- `AutoDeep/scripts/hyperparameters.py`: reads `-hp` files (text "name value", JSON, or a run's config.json) without running code (A3)
- `AutoDeep/scripts/boosted_forest_inference.py`: selects the model's features by name; `--model PATH` (default: bundled model)
- `AutoDeep/scripts/boosted_forest_training.py`: `AutoDeep train` (Ray Tune + OptunaSearch). Writes only to `--out_dir` (default `AutoDeepRun/models/<timestamp>/`): miRNA_model.json, config.json, metrics.json, user_data.csv, tuning log. `-t` takes several files; `--seed` (default 42) seeds splits, XGBoost and the Optuna sampler. `load_training_set()` and `tuning_search_alg()` are shared with alt_model_test.py
- `AutoDeep/scripts/boosted_forest_visualization.py`: `AutoDeep visualize`; tree plots only with `--trees N|all`; `--model PATH`
- `AutoDeep/scripts/alt_model_test.py`: SVM/KNN/MLP comparison for Table 2 (A4). Run as `python alt_model_test.py -m svm|knn|mlp` from an AutoDeepRun folder; same data, seed and run-folder outputs as `AutoDeep train` (`<model>_model.pkl` via joblib). Folds into evaluate.py at C1
- `AutoDeep/training_data_pre/`: 2,907 labelled loci in 11 CSVs from 8 species; the shipped model's training data, read by `AutoDeep train`
- `AutoDeep/training_data/`: 3,180 loci in 13 `new_data_*` CSVs with entropy-based 5' features (`normalized_entropy`, `normalized_entropy_s`); not used by the pipeline or shipped model
- `AutoDeep/model_weights/miRNA_model.json`: shipped model (XGBoost 2.1.1, 983 rounds x 3 classes, depth 5)
- `AutoDeep/AutoDeep.yml`: portable conda spec and the only requirements list (no requirements.txt); `AutoDeep/AutoDeep.linux-64.lock.yml`: exact Linux env the shipped model was trained in
- `review/`: REVISIONS.md, reanalysis.py (reviewer's evaluation script), reanalysis_baseline.txt (its output on commit 133bf14), reanalysis_osx-arm64.txt (its output on Tejus's Mac), METHODS_CHANGES.md
- `tests/`: pytest suite (84 tests); `conftest.py` puts `AutoDeep/scripts` on the import path
- `examples/mirdeep2_tutorial/`: miRDeep2's tutorial output (`mirdeep2_output/`) and AutoDeep's expected `xgboost_judgement.csv`; checked by `tests/test_example.py`. Regenerate the expected file when a Phase B item changes a feature on purpose
- `.github/workflows/tests.yml`: CI, pytest on ubuntu-latest and macos-latest in an env built from `AutoDeep.yml`, on pushes and PRs to `refactoring` and `main`

## Environment

- Conda env `AutoDeep` from `AutoDeep/AutoDeep.yml` (works on osx-arm64 and linux-64): Python 3.9, numpy 1.23.5, xgboost 2.1.1, ViennaRNA 2.7.0, ray 2.37, optuna 4.1, pandas 2.2.2, scikit-learn 1.5.1, pdf2image 1.17.0 (exact pin required), pytest. No pypdf.
- On Tejus's Mac the env is not activated in Claude's shell: call `/Users/tkoti/miniconda3/envs/AutoDeep/bin/python` directly, and put that `bin/` on PATH when running `AutoDeep.sh` (RNAfold and graphviz's `dot` live there).
- Tests: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests` from the repo root. Set `AUTODEEP_TEST_RUN` to the local Pomonella test run (path in CLAUDE.local.md) to include the end-to-end tests; without it they are skipped (as in CI).
- A separate conda env `mirdeep2` (miRDeep2 2.0.1.3 from bioconda) is on Tejus's Mac, used to make the tutorial example; see `examples/mirdeep2_tutorial/README.md`.
- Reanalysis on this Mac differs slightly from `reanalysis_baseline.txt` in the XGBoost rows only (arm64 vs x86 arithmetic; run-to-run identical). Compare later runs with `review/reanalysis_osx-arm64.txt`.
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
