# AutoDeep

AutoDeep is a machine learning tool that stratifies mirDeep2 outputs into Candidate, Confident, or Falsepositive labels using a gradient boosted forest approach.

## Installation

### Prerequisites

- Python 3.6 or higher
- pandas (`pip install pandas`)
- ViennaRNA (for RNAfold command)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AutoDeep
   ```

2. **Add to PATH:**
   ```bash
   export PATH="$PATH:$(pwd)"
   ```

   To make this permanent, add the following line to your shell configuration file (`~/.bashrc`, `~/.zshrc`, or equivalent):
   ```bash
   export PATH="$PATH:/path/to/AutoDeep"
   ```

3. **Verify installation:**
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
└── pdfs_<N>/
    ├── <loci_name_N>.pdf
    └── ...
```

**Output:** `AutoDeepRun/` directory containing classified miRNAs

### Training a Custom Model

Train a new model on your dataset:

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

The project requires:
- Python 3.6+
- pandas
- ViennaRNA (RNAfold)

Install Python dependencies:
```bash
pip install pandas
```

## Project Structure

```
AutoDeep/
├── AutoDeep                      # Main executable
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
├── training_data/               # Sample training datasets
│   └── original_data_*.csv
└── README.md
```

## License

See LICENSE file for details.
