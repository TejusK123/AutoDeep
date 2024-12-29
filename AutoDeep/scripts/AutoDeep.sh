#!/bin/bash


train() {
    file="$1"
    if [[ -z "$file" ]]; then
        echo "Usage: $0 train <file> <flags>"
        exit 1
    fi

    if [[ -f "$file" ]]; then
        echo "Starting model training"
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
        python "$SCRIPT_DIR/boosted_forest_training.py" "$@"

    else
        echo "Error: File '$file' does not exist."
        exit 1
    fi
}

# Main logic
if [ -z "$1" ]; then
    echo "Usage: $0 <subcommand> <args>"
    exit 1
fi

subcommand="$1"
shift  # Remove the subcommand from arguments

case "$subcommand" in
    train)
        train "$@"
        ;;
    *)
        inp_dir="$subcommand"  # Original functionality logic
        current_dir="$PWD"
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

        python "$SCRIPT_DIR/init_data.py" "$inp_dir"

        if RNAfold --help | grep -q -- '--noPS'; then
        noPS_option="--noPS"
        else
        noPS_option="-noPS"
        fi
        awk -F',' 'NR>1 {print ">" $1 "\n" $16}' "$current_dir/AutoDeepRun/formatted_novel_miRNA.csv" | RNAfold $noPS_option > "$current_dir/AutoDeepRun/RNAfold_novel_precursor_miRNAs.txt"

        for file in "$current_dir"/pdfs*/*; do
            new_file="${file//,/}"
            if [[ "$file" != "$new_file" ]]; then
                mv "$file" "$new_file"
            fi
        done

        python "$SCRIPT_DIR/csv_feature_extraction.py"
        python "$SCRIPT_DIR/folding_properties.py" "$inp_dir"

        echo "Proceeding with XGBclassifier inference"
        python "$SCRIPT_DIR/boosted_forest_inference.py"

        echo "AutoDeep Finished: Outputs in AutoDeepRun"
        ;;
esac

