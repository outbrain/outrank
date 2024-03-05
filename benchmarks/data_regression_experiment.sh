#!/bin/bash

set -euo pipefail  # Enable strict mode for safety

# Configurable variables
NUM_ROWS=1000000
NUM_FEATURES=100
INPUT_FILE="test_data_synthetic/data.csv"
SIZES=('50000' '100000' '200000' '500000' '600000' '700000' '800000' '900000' '1000000')

# Function to remove a directory safely
remove_directory_safely() {
    directory_to_remove=$1
    if [ -d "$directory_to_remove" ]; then
        echo "Removing directory: $directory_to_remove"
        rm -rvf "$directory_to_remove"
    else
        echo "Directory does not exist, skipping: $directory_to_remove"
    fi
}

# Function to generate random data
generate_data() {
    echo "Generating random data files with $NUM_ROWS rows and $NUM_FEATURES features..."
    outrank --task data_generator --num_synthetic_rows $NUM_ROWS --num_synthetic_features $NUM_FEATURES
    echo "Random data generation complete."
}

# Function to create subspaces from the data
sample_subspaces() {
    for i in "${SIZES[@]}"
    do
        dataset="test_data_synthetic/dataset_$i"
        outfile="$dataset/data.csv"
        mkdir -p "$dataset"

        if [ -f "$INPUT_FILE" ]; then
            echo "Sampling $i rows into $outfile..."
            head -n $i "$INPUT_FILE" > "$outfile"
            echo "Sampling for $outfile done."
        else
            echo "Input file $INPUT_FILE not found. Skipping sampling for $i rows."
        fi
    done
}

# Function to perform feature ranking
feature_ranking() {
    for i in "${SIZES[@]}"
    do
        dataset="test_data_synthetic/dataset_$i"
        output_folder="./test_data_synthetic/ranking_$i"

        if [ ! -d "$dataset" ]; then
            echo "Dataset directory $dataset does not exist. Skipping ranking for $i rows."
            continue
        fi

        echo "Proceeding with feature ranking for $i rows..."
        outrank --task ranking --data_path "$dataset" --data_source csv-raw \
                --combination_number_upper_bound 60 --output_folder "$output_folder" \
                --disable_tqdm True

        echo "Feature ranking summary for $i rows."
        outrank --task ranking_summary --output_folder "$output_folder" --data_path "$dataset"
        echo "Ranking for $i done."
    done
}

# Function to analyze the rankings
analyse_rankings() {
    echo "Analyzing the rankings..."
    python analyse_rankings.py test_data_synthetic
    echo "Analysis complete."
}

# Main script execution
remove_directory_safely test_data_synthetic/
generate_data
sample_subspaces
feature_ranking
analyse_rankings

echo "Script execution finished."
