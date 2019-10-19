#1/bin/bash

output_dir="additive_noise_simulation_output"
num_runs=10

mkdir -p $output_dir

categories=( "DNA_ECOLI" "Yeast_inliquid HK" "DNA_INLIQUIDDNA" "DNA_DNA@Anod" "Yeast_inliquid Live" )
encoding_schemes=( "multiplication" "trigram" "convolution" )
noise_levels=( "0_025" "0_050" "0_10" )

for category in "${categories[@]}"
do
    for level in "${noise_levels[@]}"
    do
        for scheme in "${encoding_schemes[@]}"
        do
            python hdc_model_cv.py "datasets/noisyDataSets/additive/noisy_dataset_${level}.csv" "$category" $scheme 4 $num_runs "${output_dir}/${category}_${scheme}_${level}.txt"
        done
    done
done


