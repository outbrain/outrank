##########################################################################################################
# A very generic OutRank invocation (default). It includes visualizations and other relevant statistics. #
##########################################################################################################

# This run compares features "one-at-a-time" and summarizes, visualizes the outputs.
# hint - if unsure what parameters do, you can always run "outrank --help"

outrank \
    --task all \
    --data_path $PATH_TO_YOUR_DATA \
    --data_source csv-raw \
    --heuristic MI-numba-randomized \
    --subfeature_mapping f12->f32;f1<->f41 \
    --target_ranking_only True \
    --combination_number_upper_bound 2048 \
    --num_threads 12 \
    --output_folder ./some_output_folder \
    --subsampling 10
