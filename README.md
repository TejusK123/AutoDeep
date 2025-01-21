# AutoDeep 'README'
## About

Authors: Manitejus Kotikalapudi

AutoDeep is a software that further stratifies miRDeep2 (Mackowiak, Friedländer) outputs into Candidate, Confident, and Potential False-Positive Labels


## Requirements

Linux system, Conda Environment generated from AutoDeep/AutoDeep.yml file

## Installation

### Option 1 (Only option currently)

Download the AutoDeep Directory.
Within the Directory, and with the AutoDeep conda environment activated, type

```sh
pip install -e .
```

##Script Reference

### AutoDeep 

#### Description

Core Logic of AutoDeep package. 
Formats miRDeep2 output CSV file, performs feature extraction, and then Classifies pre-miRNA loci with XGBoost

#### Input 

* Directory in which miRDeep2 was run

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

### AutoDeep visualize

#### Description

Visualizes XGBoost model via tree structure and gain metrics.


#### Input 
N/A

#### Output
* Folder containing relevant figures as png

