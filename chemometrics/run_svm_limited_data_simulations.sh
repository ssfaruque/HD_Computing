#1/bin/bash

output_dir="svm_limited_data_simulation_output"
num_runs=10

mkdir -p $output_dir

categories=( "DNA_ECOLI" "Yeast_inliquid HK" "DNA_INLIQUIDDNA" "DNA_DNA@Anod" "Yeast_inliquid Live" )
files=( 1 3 5 )

for category in "${categories[@]}"
do
    for file in "${files[@]}"
    do
        python svm_model.py datasets/our_aggregate_data.csv $file "$category" "${output_dir}/${category}_${file}.txt"
    done
done


