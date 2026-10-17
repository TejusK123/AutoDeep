# AutoDeep

AutoDeep is a machine learning tool that stratifies mirDeep2 outputs into Candidate, Confident, or Falsepositive labels using a gradient boosted forest approach.

## Installation

### Prerequisites

- Linux (x86-64) or macOS (Apple silicon)
- Conda

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TejusK123/AutoDeep.git
   cd AutoDeep/AutoDeep
   ```

2. **Create and activate the conda environment** (see [Dependencies](#dependencies)):
   ```bash
   conda env create -f AutoDeep.yml
   conda activate AutoDeep
   ```

3. **Add to PATH:**
   ```bash
   export PATH="$PATH:$(pwd)"
   ```

   To make this permanent, add the following line to your shell configuration file (`~/.bashrc`, `~/.zshrc`, or equivalent):
   ```bash
   export PATH="$PATH:/path/to/AutoDeep"
   ```

4. **Verify installation:**
   ```bash
   AutoDeep --help
   ```

## Usage

### Basic Inference

Run AutoDeep on a directory containing mirDeep2 outputs:

```bash
AutoDeep <directory>
```

**Input:** Directory with the following structure:
```
directory/
├── result_<N>.csv
└── output.mrd
```

**Output:** `AutoDeepRun/` directory containing classified miRNAs

### Training a Custom Model

Train a new model on your dataset. The model, its config, metrics and your labelled data are written to `AutoDeepRun/models/<timestamp>/`; the bundled model is never changed. Use a trained model with `AutoDeep infer <directory> --model <path>`.

```bash
AutoDeep train [options]
```

For detailed options:
```bash
AutoDeep train --help
```

### Visualizing Results

Generate visualizations for model analysis:

```bash
AutoDeep visualize [options]
```

For detailed options:
```bash
AutoDeep visualize --help
```

## Dependencies

All dependencies are listed in `AutoDeep.yml`: Python 3.9, pandas, numpy, scikit-learn, XGBoost, Ray Tune and Optuna (training), ViennaRNA (the `RNAfold` command), graphviz (tree plots) and their versions. Some are command-line programs that pip cannot install, so use conda. `AutoDeep.linux-64.lock.yml` is the exact Linux environment the bundled model was trained in.

## Project Structure

```
AutoDeep/
├── AutoDeep                      # Main executable
├── AutoDeep.yml                  # Conda environment (all dependencies)
├── scripts/
│   ├── AutoDeep.sh              # Core logic dispatcher
│   ├── init_data.py             # Data initialization
│   ├── csv_feature_extraction.py
│   ├── folding_properties.py    # RNAfold integration
│   ├── boosted_forest_inference.py
│   ├── boosted_forest_training.py
│   ├── boosted_forest_visualization.py
│   └── ...
├── model_weights/
│   └── miRNA_model.json
├── training_data_pre/           # Training data of the shipped model (used by AutoDeep train)
│   └── original_data_*.csv
├── training_data/               # Newer data with entropy-based 5' features (not yet used)
│   └── new_data_*.csv
└── README.md
```

## License

GNU General Public License v3.0; see the LICENSE file.
