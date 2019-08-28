#1/bin/bash

output_dir="split_simulation_output"
num_runs=10

mkdir -p $output_dir

categories=( "DNA_ECOLI" "Yeast_inliquid HK" "DNA_INLIQUIDDNA" "DNA_DNA@Anod" "Yeast_inliquid Live" )
encoding_schemes=( "multiplication" "trigram" "convolution" )
splits=( 2 3 4 5 )

for category in "${categories[@]}"
do
    for split in "${splits[@]}"
    do
        for scheme in "${encoding_schemes[@]}"
        do
            python hdc_model.py datasets/our_aggregate_data.csv $category $scheme $split $num_runs "${output_dir}/${category}_${scheme}_${split}.txt"
        done
    done
done


