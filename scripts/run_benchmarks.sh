
pip install . --upgrade;
cd benchmarks;

###################################################################
#..................................................................
###################################################################
# Can we find a needle

if [[ $1 == "CI" ]]
then
    echo "CI Run experiments initialized"
    # Generate relevant synthetic data sets
    python generator_naive.py --output_df_name dataset_naive --num_features 100 --size 10000;

    # Substantial subsampling must retrieve the needle.
    outrank --data_path dataset_naive --data_source csv-raw --subsampling 1 --task all --heuristic MI-numba-randomized --target_ranking_only True --interaction_order 1 --output_folder ./ranking_outputs --minibatch_size 20000;

    python generator_naive.py --verify_outputs ranking_outputs;

    rm -r ranking_outputs dataset_naive;
    exit
fi
###################################################################
#..................................................................
###################################################################
# Can we find a needle - bigger data set

# Generate relevant synthetic data sets
python generator_naive.py --output_df_name dataset_naive --num_features 300 --size 2000000;

# Substantial subsampling must retrieve the needle.
outrank --data_path dataset_naive --data_source csv-raw --subsampling 100 --task all --heuristic MI-numba-randomized --target_ranking_only True --interaction_order 1 --output_folder ./ranking_outputs --minibatch_size 20000;

python generator_naive.py --verify_outputs ranking_outputs;

rm -r ranking_outputs dataset_naive;

###################################################################
#..................................................................
###################################################################
# Can we find an interaction needle?

# Generate relevant synthetic data sets
python generator_second_order.py --output_df_name dataset_naive --num_features 100 --size 10000;

# Substantial subsampling must retrieve the needle.
outrank --data_path dataset_naive --data_source csv-raw --subsampling 1 --task all --heuristic MI-numba-randomized --target_ranking_only True --interaction_order 2 --output_folder ./ranking_outputs;

python generator_second_order.py --verify_outputs ranking_outputs;

rm -r ranking_outputs dataset_naive;

###################################################################
#..................................................................
###################################################################
# Can we find an interaction needle - order 3 with samplied stream

# Generate relevant synthetic data sets
python generator_third_order.py --output_df_name dataset_naive --num_features 100 --size 100000;

# Substantial subsampling must retrieve the needle.
outrank --data_path dataset_naive --data_source csv-raw --subsampling 10 --task all --heuristic MI-numba-randomized --target_ranking_only True --interaction_order 3 --output_folder ./ranking_outputs;

python generator_third_order.py --verify_outputs ranking_outputs;

rm -r ranking_outputs dataset_naive;

cd ..;
