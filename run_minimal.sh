
DATA_ENDPOINT="https://raw.githubusercontent.com/shenweichen/DeepCTR/master/examples/avazu_sample.txt"
pip install . --upgrade;
cd benchmarks;

###################################################################
#..................................................................
###################################################################
# Can we find a needle

# Generate relevant synthetic data sets
rm -r ranking_outputs; rm -r dataset_naive;
wget $DATA_ENDPOINT
cat avazu_sample.txt avazu_sample.txt avazu_sample.txt > tmp.txt; mv tmp.txt avazu_sample.txt; rm -rf tmp.txt;
mkdir avazu; mv avazu_sample.txt avazu/data.csv;rm -rf avazu_sample.csv;

# Substantial subsampling must retrieve the needle.
outrank --data_path avazu --data_source csv-raw --subsampling 1 --task all --heuristic MI-numba-3mr --target_ranking_only False --interaction_order 1 --output_folder ./ranking_outputs --minibatch_size 100 --label click;

echo "Ranking outputs are present in benchmarks/ranking_outputs .."
#rm -rf ranking_outputs dataset_naive;
ls ranking_outputs;

cd ..;
