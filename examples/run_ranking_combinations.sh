##########################################################################################################
# Ranking of feature combinations
##########################################################################################################

# This run demonstrates how to perform "supervised combination ranking" - the process of figuring out
# which feature combinations are potentially promising.
# Note that this process' time is directly correlated with interaction order (higher=longer runs)

# hint - if unsure what parameters do, you can always run "outrank --help"
# Example for feature pairs
outrank \
    --task all \
    --data_path $PATH_TO_YOUR_DATA \
    --data_source csv-raw \
    --heuristic MI-numba-randomized \
    --target_ranking_only True \
    --interaction_order 2 \
    --combination_number_upper_bound 2048 \
    --num_threads 50 \
    --output_folder ./some_output_folder \
    --subsampling 100


# And feature triplets. The combination_number_upper_bound bounds the number of sampled combinations (RAM controller)
outrank \
    --task all \
    --data_path $PATH_TO_YOUR_DATA \
    --data_source csv-raw \
    --heuristic MI-numba-randomized \
    --target_ranking_only True \
    --interaction_order 3 \
    --combination_number_upper_bound 2048 \
    --num_threads 50 \
    --output_folder ./some_output_folder_triplets \
    --subsampling 100
