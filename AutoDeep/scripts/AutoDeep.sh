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
                        Should contain: result_*.csv and pdfs_*/ subdirectories

OPTIONS:
    -h, --help          Show this help message

OUTPUT:
    Results are saved to:
    - AutoDeepRun/                          (created in current working directory)
    - AutoDeepRun/formatted_novel_miRNA.csv (processed miRNA data)
    - AutoDeepRun/AutoDeep_results.csv      (classification results)

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

OPTIONS:
    -h, --help                  Show this help message
    -t, --targets_path <path>   Path to CSV file with custom training targets
                                (CSV format: column 1 = loci names, column 2 = class)
                                Valid classes: Candidate, Confident, falsepositive
    -n, --no_db_data            Omit original database data from training
    -r, --tuning_rounds <int>   Number of hyperparameter tuning rounds (default: 10)
    -o, --output <name>         Output training log file name (default: training_log)
    -nw, --no_weights           Skip saving model weights (recommended for testing)
    -hp, --hyperparameters <path>
                                Path to hyperparameter config file for manual tuning
                                (overrides automatic tuning)

TARGET FILE FORMAT:
    CSV with two columns:
    - Column 1: loci names (must match fully_formatted_data.csv)
    - Column 2: class labels (Candidate, Confident, or falsepositive)

HYPERPARAMETER CONFIG FORMAT:
    Text file with one parameter per line, format: name value
    Valid parameters: max_depth, min_child_weight, subsample, eta, n_estimators,
                      gamma, base_score, alpha, lambda, colsample_bytree,
                      colsample_bylevel, colsample_bynode
    Example:
        max_depth 6
        eta 0.1
        n_estimators 500

OUTPUT:
    - model_weights/miRNA_model.json  (trained model)
    - training_log.csv                (training metrics, if using auto-tuning)

EXAMPLES:
    AutoDeep train
    AutoDeep train --targets_path labels.csv --tuning_rounds 1000
    AutoDeep train --targets_path labels.csv --hyperparameters config.txt
    AutoDeep train --no_db_data --targets_path custom_labels.csv

EOF
}

# Display visualize command help
show_visualize_help() {
    cat << 'EOF'
AutoDeep visualize - Analyze and visualize model results

USAGE:
    AutoDeep visualize [OPTIONS]

DESCRIPTION:
    Generates visualizations and analyses of the trained model including:
    - Feature importance plots (by gain, weight, and cover)
    - Individual decision tree visualizations

REQUIREMENTS:
    - Must be run from within an AutoDeepRun directory
    - Model must exist: model_weights/miRNA_model.json
    - Requires graphviz system package for tree rendering

OPTIONS:
    -h, --help              Show this help message
    --no_tree               Skip generating individual decision tree plots (faster)
    -o, --output <dir>      Output directory base name (default: tree_plots)
                            Full path will be <name><timestamp>/

OUTPUT:
    Visualizations are saved in timestamped directory:
    - feature_importance_gain.png       (feature importance by gain)
    - feature_importance_weight.png     (feature importance by weight)
    - feature_importance_cover.png      (feature importance by cover)
    - tree_plot0.png, tree_plot1.png... (individual decision trees, if not --no_tree)

EXAMPLES:
    cd AutoDeepRun
    AutoDeep visualize
    AutoDeep visualize --no_tree
    AutoDeep visualize -o my_plots

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
    
    if [[ -z "$input_dir" ]]; then
        error "Missing required argument: directory"
    fi
    
    if [[ ! -d "$input_dir" ]]; then
        error "Input directory does not exist: $input_dir"
    fi
    
    info "Starting inference pipeline for: $input_dir"
    
    local current_dir="$PWD"
    
    # Step 1: Initialize and format data
    info "Step 1/5: Initializing data..."
    python "$SCRIPT_DIR/init_data.py" "$input_dir" || error "Data initialization failed"
    
    # Step 2: Detect correct RNAfold option format for this system
    info "Step 2/5: Detecting RNAfold version..."
    local noPS_option="-noPS"
    if RNAfold --help 2>/dev/null | grep -q -- '--noPS'; then
        noPS_option="--noPS"
        info "Using RNAfold option: $noPS_option"
    fi
    
    # Step 3: Extract sequences and run RNAfold
    info "Step 3/5: Running RNAfold for secondary structure prediction..."
    if [[ ! -f "$current_dir/AutoDeepRun/formatted_novel_miRNA.csv" ]]; then
        error "Required file not found: AutoDeepRun/formatted_novel_miRNA.csv"
    fi
    
    awk -F',' 'NR>1 {print ">" $1 "\n" $16}' "$current_dir/AutoDeepRun/formatted_novel_miRNA.csv" | \
        RNAfold "$noPS_option" > "$current_dir/AutoDeepRun/RNAfold_novel_precursor_miRNAs.txt" || \
        error "RNAfold execution failed"
    
    # Step 4: Clean up filenames (remove commas which cause issues)
    info "Step 4/5: Cleaning up filenames..."
    if [[ -d "$current_dir/pdfs_"* ]]; then
        while IFS= read -r -d '' file; do
            local new_file="${file//,/}"
            if [[ "$file" != "$new_file" ]]; then
                mv "$file" "$new_file"
            fi
        done < <(find "$current_dir/pdfs_"* -type f -print0 2>/dev/null || true)
    fi
    
    #TODO: Do the same filename cleanup for .mrd file


    # Step 5: Extract features and run classification
    info "Step 5/5: Extracting features and running classification..."
    python "$SCRIPT_DIR/csv_feature_extraction.py" || error "Feature extraction failed"
    python "$SCRIPT_DIR/folding_properties.py" "$input_dir" || error "Folding properties analysis failed"
    python "$SCRIPT_DIR/boosted_forest_inference.py" || error "Classification inference failed"
    
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
    
    # Check if we're in an AutoDeepRun directory
    if [[ ! -d "AutoDeepRun" ]]; then
        error "This command must be run from a directory containing 'AutoDeepRun/'"
    fi
    
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

