##########################################################################################################
# Pairwise feature ranking (feature redundancy calculation)
##########################################################################################################

# This run demonstrates how to obtain "feature heatmaps" - pairwise summaries of mutual redundancy
# Note that pairwise calculations take more time - increasing thread count is a possible mitigation

# hint - if unsure what parameters do, you can always run "outrank --help"
outrank \
    --task all \
    --data_path $PATH_TO_YOUR_DATA \
    --data_source csv-raw \
    --heuristic MI-numba-randomized \
    --target_ranking_only False \
    --combination_number_upper_bound 2048 \
    --num_threads 50 \
    --output_folder ./some_output_folder \
    --subsampling 100
