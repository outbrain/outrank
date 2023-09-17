##########################################################################################################
# Performing non-myopic ranking with 3MR
##########################################################################################################

# This run computes 3MR-based rankings, see repo's papers for more details.
# hint - if unsure what parameters do, you can always run "outrank --help"

outrank \
    --task all \
    --data_path $PATH_TO_YOUR_DATA \
    --data_source csv-raw \
    --heuristic MI-numba-3mr \
    --target_ranking_only True \
    --combination_number_upper_bound 2048 \
    --num_threads 12 \
    --interaction_order 1 \
    --transformers fw-transformers \
    --output_folder ./some_output_folder \
    --subsampling 30
