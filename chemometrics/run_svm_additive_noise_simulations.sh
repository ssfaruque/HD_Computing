#1/bin/bash

output_dir="svm_additive_noise_simulation_output"
num_runs=10

mkdir -p $output_dir

categories=( "DNA_ECOLI" "Yeast_inliquid HK" "DNA_INLIQUIDDNA" "DNA_DNA@Anod" "Yeast_inliquid Live" )
noise_levels=( "0_025" "0_050" "0_10" )

for category in "${categories[@]}"
do
    for level in "${noise_levels[@]}"
    do
        python svm_model_cv.py "datasets/noisyDataSets/additive/noisy_dataset_${level}.csv" 4 "$category" "${output_dir}/${category}_${level}.txt"
    done
done


