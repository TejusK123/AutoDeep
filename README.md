# AutoDeep 'README'
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
* XGBoost label prediction CSV

#### Flags
 * --model <path>                Model JSON to use (default: the bundled model)

### AutoDeep train

#### Description

Trains a new XGBoost model with user data. Must be run within the AutoDeepRun directory.
The bundled model and training data are never changed; everything is written to the output directory.

#### Input 
* CSV file(s) with loci names in first column and class names in second, or user_data.csv from an earlier training run

#### Output
In `--out_dir` (default `AutoDeepRun/models/<timestamp>/`):
* miRNA_model.json: the trained model, for `AutoDeep infer <directory> --model <path>`
* config.json: hyperparameters, options and data used
* metrics.json: accuracy and F1 scores
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
 * --help                        Show this message and exit.
### AutoDeep visualize

#### Description

Visualizes XGBoost model via tree structure and gain metrics.


#### Input 
N/A

#### Output
* Folder containing relevant figures as png

#### Flags 
 * --no_tree          Do not output tree plots
  * -o, --output TEXT  Output directory for tree plots
  * --model <path>     Model JSON to visualize (default: the bundled model)
  * --help             Show this message and exit.

