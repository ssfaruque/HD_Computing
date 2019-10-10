#1/bin/bash

output_dir="noise_simulation_output"
num_runs=10

mkdir -p $output_dir

categories=( "DNA_ECOLI" "Yeast_inliquid HK" "DNA_INLIQUIDDNA" "DNA_DNA@Anod" "Yeast_inliquid Live" )
#encoding_schemes=( "multiplication" "trigram" "convolution" )
encoding_schemes=( "multiplication" "trigram" "convolution" )
noise_levels=( "05" "15" )

for category in "${categories[@]}"
do
    for level in "${noise_levels[@]}"
    do
        for scheme in "${encoding_schemes[@]}"
        do
            python hdc_model.py "datasets/noisyDataSets/noisy_dataset_std_0_${level}.csv" "$category" $scheme 4 $num_runs "${output_dir}/${category}_${scheme}_0_${level}.txt"
        done
    done
done


