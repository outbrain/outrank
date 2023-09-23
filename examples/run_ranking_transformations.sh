##########################################################################################################
# Ranking of feature transformations
##########################################################################################################

# A common and very important task is figuring out which transformations of a feature are promising.

# hint - if unsure what parameters do, you can always run "outrank --help"
# Example considering some generic transformations of features. Note that OutRank is type aware, if using formats such as ob-vw or ob-csv,
# type-aware transformations can be produced. See e.g., https://outbrain.github.io/outrank/outrank/algorithms/importance_estimator.html?search=ob-vw for more details on the format.
outrank \
    --task all \
    --data_path $PATH_TO_YOUR_DATA \
    --data_source csv-raw \
    --heuristic MI-numba-randomized \
    --target_ranking_only True \
    --combination_number_upper_bound 2048 \
    --num_threads 50 \
    --transformers default \
    --output_folder ./some_output_folder \
    --subsampling 100
