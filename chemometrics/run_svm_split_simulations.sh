#1/bin/bash

output_dir="svm_split_simulation_output"
num_runs=10

mkdir -p $output_dir

categories=( "DNA_ECOLI" "Yeast_inliquid HK" "DNA_INLIQUIDDNA" "DNA_DNA@Anod" "Yeast_inliquid Live" )
splits=( 2 3 4 5 )

for category in "${categories[@]}"
do
    for split in "${splits[@]}"
    do
        python svm_model.py datasets/our_aggregate_data.csv $split "$category" "${output_dir}/${category}_${split}.txt"
    done
done


