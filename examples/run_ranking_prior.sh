##########################################################################################################
# A very generic OutRank invocation (default). It includes visualizations and other relevant statistics. #
##########################################################################################################

# This run compares features "one-at-a-time" and summarizes, visualizes the outputs.
# hint - if unsure what parameters do, you can always run "outrank --help"

outrank \
    --task all \
    --data_path $PATH_TO_YOUR_DATA \
    --data_source ob-csv \
    --heuristic surrogate-SGD-prior \
    --target_ranking_only True \
    --interaction_order 1 \
    --combination_number_upper_bound 2048 \
    --num_threads 12 \
    --output_folder ./some_output_folder \
    --subsampling 1 \
    --minibatch_size 10000 \
    --label_column info_click_valid \
    --reference_model_JSON $PATH_TO_YOUR_REFERENCE_MODEL
