# AutoDeep 'README'

[![tests](https://github.com/TejusK123/AutoDeep/actions/workflows/tests.yml/badge.svg?branch=refactoring)](https://github.com/TejusK123/AutoDeep/actions/workflows/tests.yml)

## About

Authors: Manitejus Kotikalapudi

AutoDeep is a tree-boosting software that further stratifies miRDeep2 (Mackowiak, Friedländer) outputs into Candidate, Confident, and Potential False-Positive Labels


## Requirements

Linux (x86-64) or macOS (Apple silicon), and Conda

## Installation

### Option 1 (Only option currently)

Download the repository. From its root, create and activate the conda environment, then put AutoDeep on your PATH (see AutoDeep/README.md):

```sh
conda env create -f AutoDeep/AutoDeep.yml
conda activate AutoDeep
export PATH="$PATH:$(pwd)/AutoDeep"
```

`AutoDeep/AutoDeep.linux-64.lock.yml` is the exact Linux environment the shipped model was trained in.

## Example

`examples/mirdeep2_tutorial/` holds the output of miRDeep2's own tutorial run and the AutoDeep output expected for it. To try AutoDeep on it (copy the folder first: AutoDeep rewrites `result_*.csv` in place):

```sh
cp -r examples/mirdeep2_tutorial/mirdeep2_output /tmp/tutorial_run
AutoDeep infer /tmp/tutorial_run
diff AutoDeepRun/xgboost_judgement.csv examples/mirdeep2_tutorial/expected/xgboost_judgement.csv
```

See `examples/mirdeep2_tutorial/README.md` for how the miRDeep2 output was made.

## Script Reference

### AutoDeep 

#### Description

Core Logic of AutoDeep package. 
Formats miRDeep2 output CSV file, performs feature extraction, and then Classifies pre-miRNA loci with XGBoost

#### Input 

* Directory in which miRDeep2 was run

The input directory should have the following file structure:

    ├── directory in which miRDeep2 was run
    │   ├── result_<[0-9]>.csv
    │   ├── output.mrd

#### Output
A directory named "AutoDeepRun" which contains
* xgboost_judgement.csv: each locus's label, class probabilities and its three largest SHAP contributions to that label (`shap_top1_feature`, `shap_top1_value`, ...; positive values push towards the label)

#### Flags
 * --model <path>                Model JSON to use (default: the bundled model)

### AutoDeep train

#### Description

Trains a new XGBoost model with user data. Must be run within the AutoDeepRun directory.
Each set of hyperparameters (tuned with Ray Tune and Optuna, or given with `-hp`) is scored on the same ten stratified 80/20 splits, with the classes weighted equally; early stopping on a held-out fifth of each 80% chooses the number of trees unless `-hp` sets `n_estimators`. The saved model is the chosen set refit on all rows, weighted the same way.
The bundled model and training data are never changed; everything is written to the output directory.

#### Input 
* CSV file(s) with loci names in first column and class names in second, or user_data.csv from an earlier training run

#### Output
In `--out_dir` (default `AutoDeepRun/models/<timestamp>/`):
* miRNA_model.json: the trained model, for `AutoDeep infer <directory> --model <path>`
* config.json: hyperparameters (including the number of trees), seed, options, data used, and how the model was scored and fit
* metrics.json: mean and SD of accuracy and F1 scores over the ten splits
* user_data.csv: your labelled loci (pass it to `-t` to reuse them)
* training_log.csv: tuning log (automatic tuning only)

#### Flags
 * -t, --targets_path <str> [<str> ...]  Path(s) to targets files
 * -n, --no_db_data              Flag that omits original dataset from training
 * -r, --tuning_rounds <int>     Number of tuning rounds: Default <10>
 * -o, --output <str>            Name of the tuning log file
 * -nw, --no_weights             Flag that skips saving the model
 * --out_dir <path>              Output directory: Default models/<timestamp>
 * -hp, --hyperparameters <str>  Path to hyperparameter configuration file in
                                case of manual tuning: "name value" lines,
                                a JSON object, or config.json from an earlier
                                run (see AutoDeep train --help)
 * --seed <int>                  Random seed for data splits, XGBoost and tuning:
                                Default <42>, saved in config.json
 * --help                        Show this message and exit.
### AutoDeep visualize

#### Description

Explains the model's predictions with SHAP values (exact TreeSHAP, computed by XGBoost): one summary plot per class, the mean absolute SHAP value of each feature per class, and dependence plots for 5' homogeneity, the two overhangs and folding energy. The plots are vector files (PDF or SVG) with 8 pt Helvetica text kept as text.


#### Input 
The bundled training data, or the CSV(s) given with `--data` (e.g. `AutoDeepRun/fully_formatted_data.csv` from `AutoDeep infer`)

#### Output
A folder `plots<timestamp>/` with
* shap_summary_<class>.pdf, shap_importance.pdf, shap_dependence_<feature>.pdf
* shap_values.csv (every locus's SHAP values) and shap_importance.csv (the numbers behind shap_importance.pdf)
* feature_importance_{gain,weight,cover}.png with `--legacy_importance`; decision tree PNGs with `--trees`

#### Flags 
 * --data <csv> [<csv> ...]  CSV(s) to explain (default: the bundled training data)
 * --format <pdf|svg>        File format of the SHAP plots (default: pdf)
 * --legacy_importance       Also plot XGBoost's feature importance by gain, weight and cover, as before
 * --trees <N|all>           Also plot the trees of the first N rounds (one per class), or all trees
 * -o, --output TEXT         Output directory base name (default: plots)
 * --model <path>            Model JSON to visualize (default: the bundled model)
 * --help                    Show this message and exit.

