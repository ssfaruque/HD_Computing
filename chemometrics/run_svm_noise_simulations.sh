#1/bin/bash

output_dir="svm_noise_simulation_output"
num_runs=10

mkdir -p $output_dir

categories=( "DNA_ECOLI" "Yeast_inliquid HK" "DNA_INLIQUIDDNA" "DNA_DNA@Anod" "Yeast_inliquid Live" )
noise_levels=( 1 2 3 )

for category in "${categories[@]}"
do
    for level in "${noise_levels[@]}"
    do
        python svm_model.py "datasets/noisyDataSets/noisy_dataset_std_0_${level}.csv" 4 "$category" "${output_dir}/${category}_0_${level}.txt"
    done
done


