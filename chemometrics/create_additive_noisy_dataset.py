import sys
import csv
import numpy as np

def load_dataset(dataset_name):
    file = open(dataset_name, "r")
    csv_reader = csv.reader(file)
    dataset = []
    for row in csv_reader:
        dataset.append(row)

    file.close()
    return dataset


def clamp(val, min_val, max_val):
    val = max(0, val)
    val = min(val, 1.0)
    return val

def create_noisy_dataset(dataset_name, constant_error, output_file_name):
    dataset = load_dataset(dataset_name)

    for i  in range(1, len(dataset)):
        for j in range(2,len(dataset[0])):
            dataset[i][j] = float(dataset[i][j]) + constant_error
            dataset[i][j] = clamp(dataset[i][j], 0.0, 1.0)


    file = open(output_file_name, "w")
    csv_writer = csv.writer(file, delimiter=",")

    for row in dataset:
        csv_writer.writerow(row)

    file.close()


def main():
    create_noisy_dataset(sys.argv[1], float(sys.argv[2]), sys.argv[3])




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 create_noisy_dataset.py name_of_dataset constant_error output_file_name")
        print("e.g. python3 create_noisy_dataset.py datasets/our_aggregate_data.csv 0.025 datasets/noisy_dataset_var_0.05.csv")
    else:    
        main()