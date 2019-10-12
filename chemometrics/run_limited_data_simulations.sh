#1/bin/bash

output_dir="limited_data_simulation_output"
num_runs=10

mkdir -p $output_dir

categories=( "DNA_ECOLI" "Yeast_inliquid HK" "DNA_INLIQUIDDNA" "DNA_DNA@Anod" "Yeast_inliquid Live" )

encoding_schemes=( "multiplication" "trigram" "convolution" )
num_files=( 1 3 5 )

for category in "${categories[@]}"
do
    for file in "${num_files[@]}"
    do
        for scheme in "${encoding_schemes[@]}"
        do
            python hdc_model.py datasets/our_aggregate_data.csv "$category" $scheme $file $num_runs "${output_dir}/${category}_${scheme}_${file}.txt"
        done
    done
done


