#!/bin/bash

################################################################################
# AutoDeep - miRNA Classifier Pipeline
# 
# A machine learning tool that stratifies mirDeep2 outputs into Candidate,
# Confident, or Falsepositive labels using a gradient boosted forest approach.
################################################################################

set -euo pipefail

# ============================================================================
# CONSTANTS
# ============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly PROGRAM_NAME="$(basename "$0")"
readonly VERSION="0.1"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Print colored error message and exit
error() {
    echo -e "${RED}Error: $*${NC}" >&2
    exit 1
}

# Print colored warning message
warn() {
    echo -e "${YELLOW}Warning: $*${NC}" >&2
}

# Print informational message
info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

# ============================================================================
# HELP FUNCTIONS
# ============================================================================

# Display general help message
show_general_help() {
    cat << 'EOF'
AutoDeep - miRNA Classification Pipeline

USAGE:
    AutoDeep <command> [options]

COMMANDS:
    infer       Run inference on a directory with mirDeep2 outputs (default)
    train       Train a new classification model
    visualize   Visualize model results and analysis
    help        Show this help message or help for a specific command

OPTIONS:
    -h, --help      Show this help message
    -v, --version   Show version information

EXAMPLES:
    AutoDeep /path/to/mirdeep2_output
    AutoDeep infer /path/to/mirdeep2_output
    AutoDeep train --help
    AutoDeep visualize --help

For more information on a specific command, run:
    AutoDeep <command> --help

EOF
}

# Display version information
show_version() {
    echo "AutoDeep version $VERSION"
}

# Display inference command help
show_infer_help() {
    cat << 'EOF'
AutoDeep infer - Run inference on mirDeep2 outputs

USAGE:
    AutoDeep infer <directory> [options]
    AutoDeep <directory>  (shorthand)

DESCRIPTION:
    Processes a directory containing mirDeep2 outputs and classifies novel
    miRNAs into Candidate, Confident, or Falsepositive categories using a
    trained gradient boosted forest model.

ARGUMENTS:
    <directory>         Path to directory containing mirDeep2 outputs
                        Should contain: result_*.csv and output.mrd

OPTIONS:
    -h, --help          Show this help message
    --model <path>      Model JSON to use (default: the bundled model),
                        e.g. one written by AutoDeep train

OUTPUT:
    Results are saved to:
    - AutoDeepRun/                          (created in current working directory)
    - AutoDeepRun/formatted_novel_miRNA.csv (processed miRNA data)
    - AutoDeepRun/xgboost_judgement.csv     (labels, class probabilities and each
                                             locus's three largest SHAP contributions
                                             to its label: shap_top1_feature,
                                             shap_top1_value, ...)

DEPENDENCIES:
    - Python 3.6+
    - pandas
    - xgboost
    - ViennaRNA (RNAfold)

EXAMPLES:
    AutoDeep /data/mirdeep2_run_2024
    AutoDeep infer /data/mirdeep2_run_2024

EOF
}

# Display train command help
show_train_help() {
    cat << 'EOF'
AutoDeep train - Train a new classification model

USAGE:
    AutoDeep train [OPTIONS]

DESCRIPTION:
    Trains a new gradient boosted forest classification model on your dataset.
    This allows customization of model parameters and optimization for
    specific organisms or datasets.

    Each set of hyperparameters (tuned, or from -hp) is scored on the same ten
    stratified 80/20 splits, with the classes weighted equally. The number of
    trees is chosen by early stopping on a held-out fifth of each 80% (at most
    3000 trees), unless -hp sets n_estimators. The saved model is the chosen
    set refit on all rows, weighted the same way.

OPTIONS:
    -h, --help                  Show this help message
    -t, --targets_path <path> [<path> ...]
                                CSV file(s) with custom training targets
                                (CSV format: column 1 = loci names, column 2 = class),
                                or user_data.csv from an earlier training run
                                Valid classes: Candidate, Confident, falsepositive
    -n, --no_db_data            Omit original database data from training
    -r, --tuning_rounds <int>   Number of hyperparameter tuning rounds (default: 10)
    -o, --output <name>         Tuning log file name (default: training_log)
    -nw, --no_weights           Skip saving the model (other outputs are still written)
    --out_dir <path>            Output directory (default: models/<timestamp>)
    -hp, --hyperparameters <path>
                                Path to hyperparameter config file for manual tuning
                                (overrides automatic tuning)
    --seed <int>                Random seed for the data splits, XGBoost and the
                                tuning sampler (default: 42; saved in config.json).
                                The same seed gives the same model. With more than
                                10 tuning rounds, results also depend on the
                                number of CPUs

TARGET FILE FORMAT:
    CSV with two columns:
    - Column 1: loci names (must match fully_formatted_data.csv)
    - Column 2: class labels (Candidate, Confident, or falsepositive)

HYPERPARAMETER CONFIG FORMAT:
    Text file with one parameter per line, format: name value
    (values are numbers), or a JSON object, or config.json from an
    earlier training run. Nothing in the file is run as code.
    Valid parameters: max_depth, min_child_weight, subsample, eta, n_estimators,
                      gamma, base_score, alpha, lambda, colsample_bytree,
                      colsample_bylevel, colsample_bynode
    Without n_estimators, early stopping chooses it. base_score is accepted
    but has no effect on a three-class model.
    Example (text):
        max_depth 6
        eta 0.1
        n_estimators 500
    Example (JSON):
        {"max_depth": 6, "eta": 0.1, "n_estimators": 500}

OUTPUT:
    Written to --out_dir (default: AutoDeepRun/models/<timestamp>/);
    nothing inside the AutoDeep installation is changed:
    - miRNA_model.json   (trained model; use with AutoDeep infer --model)
    - config.json        (hyperparameters incl. the number of trees, seed,
                          options, data used, how the model was scored and fit)
    - metrics.json       (mean and SD of accuracy and F1 scores over the ten splits)
    - user_data.csv      (your labelled loci, if -t was given)
    - training_log.csv   (tuning log, if using auto-tuning)

EXAMPLES:
    AutoDeep train
    AutoDeep train --targets_path labels.csv --tuning_rounds 1000
    AutoDeep train --targets_path labels.csv --hyperparameters hp.json
    AutoDeep train --no_db_data --targets_path custom_labels.csv

EOF
}

# Display visualize command help
show_visualize_help() {
    cat << 'EOF'
AutoDeep visualize - Explain the model's predictions with SHAP values

USAGE:
    AutoDeep visualize [OPTIONS]

DESCRIPTION:
    Computes each feature's SHAP value (its contribution to each class's score,
    exact TreeSHAP from XGBoost) for every locus in the data, and plots:
    - one summary plot per class (every locus's SHAP value for every feature)
    - the mean absolute SHAP value of each feature, per class
    - dependence plots for 5' homogeneity, the two overhangs and folding energy
    Text is 8 pt Helvetica (Arial or DejaVu Sans where Helvetica is missing),
    kept as text in the PDF or SVG.

REQUIREMENTS:
    - Requires graphviz for tree plots (included in the conda environment)

OPTIONS:
    -h, --help              Show this help message
    --data <csv> [...]      CSV(s) with the model's features to explain, e.g.
                            AutoDeepRun/fully_formatted_data.csv
                            (default: the bundled training data)
    --format <pdf|svg>      File format of the SHAP plots (default: pdf)
    --legacy_importance     Also plot XGBoost's feature importance by gain,
                            weight and cover (PNG), as before
    --trees <N|all>         Also plot the trees of the first N boosting rounds
                            (one tree per class), or every tree with 'all'
    -o, --output <dir>      Output directory base name (default: plots)
                            Full path will be <name><timestamp>/
    --model <path>          Model JSON to visualize (default: the bundled model)

OUTPUT:
    Visualizations are saved in timestamped directory:
    - shap_summary_Candidate.pdf, ...   (one summary plot per class)
    - shap_importance.pdf               (mean absolute SHAP value per class)
    - shap_dependence_<feature>.pdf     (five_prime_homogeneity, overhang_bot,
                                         overhang_top, folding_energy)
    - shap_values.csv                   (every locus's SHAP values, one row per
                                         locus and class)
    - shap_importance.csv               (the numbers behind shap_importance.pdf)
    - feature_importance_gain.png, ...  (with --legacy_importance)
    - tree_round0000_Candidate.png, ... (decision trees, with --trees)

EXAMPLES:
    AutoDeep visualize
    AutoDeep visualize --data AutoDeepRun/fully_formatted_data.csv
    AutoDeep visualize --format svg
    AutoDeep visualize --legacy_importance
    AutoDeep visualize --trees 5

TROUBLESHOOTING:
    If graphviz is not found, install it:
    - Ubuntu/Debian: sudo apt install graphviz
    - macOS: brew install graphviz
    - Conda: conda install graphviz

EOF
}

# ============================================================================
# DEPENDENCY CHECKING
# ============================================================================

check_dependencies() {
    local missing_deps=()
    
    # Check for Python
    if ! command -v python &> /dev/null; then
        missing_deps+=("python")
    fi
    
    # Check for RNAfold (only needed for inference)
    if ! command -v RNAfold &> /dev/null; then
        warn "RNAfold not found. This is required for inference command."
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        error "Missing required dependencies: ${missing_deps[*]}"
    fi
}

# ============================================================================
# COMMAND FUNCTIONS
# ============================================================================

# Infer command: Run the full inference pipeline
cmd_infer() {
    # Check for help flag
    if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
        show_infer_help
        exit 0
    fi
    
    local input_dir="$1"
    shift  # remaining options go to the inference step
    
    if [[ -z "$input_dir" ]]; then
        error "Missing required argument: directory"
    fi
    
    if [[ ! -d "$input_dir" ]]; then
        error "Input directory does not exist: $input_dir"
    fi
    
    info "Starting inference pipeline for: $input_dir"
    
    local current_dir="$PWD"
    
    # Step 1: Initialize and format data
    info "Step 1/4: Initializing data..."
    python "$SCRIPT_DIR/init_data.py" "$input_dir" || error "Data initialization failed"
    
    # Step 2: Detect correct RNAfold option format for this system
    info "Step 2/4: Detecting RNAfold version..."
    local noPS_option="-noPS"
    if RNAfold --help 2>/dev/null | grep -q -- '--noPS'; then
        noPS_option="--noPS"
        info "Using RNAfold option: $noPS_option"
    fi
    
    # Step 3: Extract sequences and run RNAfold
    info "Step 3/4: Running RNAfold for secondary structure prediction..."
    if [[ ! -f "$current_dir/AutoDeepRun/formatted_novel_miRNA.csv" ]]; then
        error "Required file not found: AutoDeepRun/formatted_novel_miRNA.csv"
    fi
    
    awk -F',' 'NR>1 {print ">" $1 "\n" $16}' "$current_dir/AutoDeepRun/formatted_novel_miRNA.csv" | \
        RNAfold "$noPS_option" > "$current_dir/AutoDeepRun/RNAfold_novel_precursor_miRNAs.txt" || \
        error "RNAfold execution failed"
    
    # Step 4: Extract features and run classification
    info "Step 4/4: Extracting features and running classification..."
    python "$SCRIPT_DIR/csv_feature_extraction.py" || error "Feature extraction failed"
    python "$SCRIPT_DIR/folding_properties.py" "$input_dir" || error "Folding properties analysis failed"
    python "$SCRIPT_DIR/boosted_forest_inference.py" "$@" || error "Classification inference failed"
    
    info "AutoDeep inference completed successfully!"
    echo ""
    echo "Results saved to: AutoDeepRun/"
}

# Train command: Train a new model
cmd_train() {
    # Check for help flag in arguments
    for arg in "$@"; do
        if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
            show_train_help
            exit 0
        fi
    done
    
    check_dependencies
    info "Starting model training..."
    python "$SCRIPT_DIR/boosted_forest_training.py" "$@" || error "Model training failed"
    info "Model training completed successfully!"
}

# Visualize command: Generate visualizations
cmd_visualize() {
    # Check for help flag in arguments
    for arg in "$@"; do
        if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
            show_visualize_help
            exit 0
        fi
    done
    
    check_dependencies

    info "Starting visualization..."
    python "$SCRIPT_DIR/boosted_forest_visualization.py" "$@" || error "Visualization failed"
    info "Visualization completed successfully!"
}

# ============================================================================
# MAIN SCRIPT LOGIC
# ============================================================================

main() {
    local command="${1:-}"
    
    case "$command" in
        "")
            # No command provided
            show_general_help
            exit 0
            ;;
        -h|--help|help)
            # Show general help or specific command help
            if [[ $# -gt 1 ]]; then
                local subcommand="$2"
                case "$subcommand" in
                    infer)
                        show_infer_help
                        ;;
                    train)
                        show_train_help
                        ;;
                    visualize)
                        show_visualize_help
                        ;;
                    *)
                        error "Unknown command: $subcommand"
                        ;;
                esac
            else
                show_general_help
            fi
            exit 0
            ;;
        -v|--version)
            show_version
            exit 0
            ;;
        infer)
            check_dependencies
            shift
            cmd_infer "$@"
            ;;
        train)
            shift
            cmd_train "$@"
            ;;
        visualize)
            shift
            cmd_visualize "$@"
            ;;
        -*)
            error "Unknown option: $command"
            ;;
        *)
            # Default to inference if first argument looks like a directory
            check_dependencies
            cmd_infer "$command"
            ;;
    esac
}

# Run main function with all arguments
main "$@"

