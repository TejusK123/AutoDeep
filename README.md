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
    │   ├── pdfs_<[0-9]>
    │   │   ├── <loci_name_[0-9]>.pdf

#### Output
A directory named "AutoDeepRun" which contains
* XGBoost label prediction CSV

### AutoDeep train

#### Description

Trains AutoDeep's underlying XGBoost model with user data.
Must be run within AutoDeepRun directory

#### Input 
* CSV file with loci names in first column and class names in second

#### Output
* Training Log CSV

#### Flags
* -n Omits original training data from model training (i.e only uses your inputs)
* -t, --targets_path <str>      Path to targets file
 * -n, --no_db_data              Flag that omits original dataset from training
 * -r, --tuning_rounds <int>     Number of tuning rounds: Default <10>
 * -o, --output <str>            Name of output training_log file
 * -nw, --no_weights             Flag that omits saving the model weights
                                (recommended for testing)
 * -hp, --hyperparameters <str>  Path to hyperparameter configuration file in
                                case of manual tuning
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
  * --help             Show this message and exit.

