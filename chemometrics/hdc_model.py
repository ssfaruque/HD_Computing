import hdc
import sys
import random as rand
import numpy as np
import math
import pickle
import time
import statistics as stats
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# dataset_name = sys.argv[1]
# category = sys.argv[2]
# scheme = sys.argv[3]
# num_splits = int(sys.argv[4])
# num_runs = int(sys.argv[5])
# output_file = sys.argv[6]


D = 10000 # Dimensionality = 10000
rand_indices = rand.sample(range(D), D // 2) # Stores indices from 0 to D-1 in random order
threshold = 0 # Initializing threshold parameter for preprocessing
threshold_values = {"DNA_ECOLI": 0.065,
                    "Yeast_inliquid HK": 0.055,
                    "DNA_INLIQUIDDNA": 0.0875,
                    "DNA_DNA@Anod": 0.07,
                    "Yeast_inliquid Live": 0.07}
category_names = {"1": "DNA_ECOLI",
                  "2": "DNA_DNA@Anod",
                  "3": "DNA_INLIQUIDDNA",
                  "4": "Yeast_inliquid HK",
                  "5": "Yeast_inliquid Live"}


# Generate the CiM hypervector corresponding to an absorbance on the fly
def calc_abs_iM(min_hv, abs_val, D, m):
    if  abs_val < threshold:
        return np.zeros(D)
    num_bits_to_flip = math.floor((D / 2) / (m - 1))
    hv = np.copy(min_hv)
    step_size = 1 / (m - 1)
    level = round(abs_val / step_size)
    hv[rand_indices[0 : (num_bits_to_flip * level)]] *= -1
    return hv


# Generate the CiM hypervector corresponding to a wavenumber on the fly
def calc_wn_iM(min_hv, index, D, m):
    num_bits_to_flip = math.floor((D / 2) / (m - 1))
    hv = np.copy(min_hv)
    level = index
    hv[rand_indices[0 : (num_bits_to_flip * level)]] *= -1
    return hv


# ==================== ENCODING SCHEMES ====================
def trigram(absorbances, min_abs_hv, min_wn_hv, D, n=3):
    start = 0
    end = n
    index = 0
    sum_hv = np.zeros(D)

    while end < len(absorbances) - n + 1:
        n_gram_abs = absorbances[start:end]
        prod_hv = np.ones(D)
        num_shifts = n - 1

        for absorbance in n_gram_abs:
            absorbance_hv = calc_abs_iM(min_abs_hv, absorbances[index], D, m=1001)
            tmp_hv = np.roll(absorbance_hv, num_shifts)
            prod_hv *= tmp_hv
            index += 1
            num_shifts -= 1

        sum_hv += prod_hv
        index -= (n - 1)
        start += 1
        end += 1

    return sum_hv


def convolution(absorbances, min_abs_hv, min_wn_hv, D):
    sum_hv = np.zeros(D)

    for i in range(0,len(absorbances)):
        absorbance_hv = calc_abs_iM(min_abs_hv, absorbances[i], D, m=1001)
        wavenum_hv = calc_wn_iM(min_wn_hv, i, D, m=(len(absorbances) + 1))
        result_hv = np.convolve(absorbance_hv, wavenum_hv, mode="same")
        sum_hv += result_hv

    return sum_hv


def multiplication(absorbances, min_abs_hv, min_wn_hv, D, n=1):
    sum_hv = np.zeros(D)

    for i in range(0,len(absorbances)):
        absorbance_hv = calc_abs_iM(min_abs_hv, absorbances[i], D, m=1001)
        wavenum_hv = calc_wn_iM(min_wn_hv, i, D, m=(len(absorbances) + 1))
        result_hv = absorbance_hv * wavenum_hv
        sum_hv += result_hv

    return sum_hv


schemes = {"c": convolution,
           "m": multiplication,
           "t": trigram}
# ============================================================

# Binarizes a hypervector to +1s and -1s
def binarizeHV(hv, threshold):
    for i in range(len(hv)):
        if hv[i] > threshold:
            hv[i] = 1
        else:
            hv[i] = -1
    return hv


# HDC model for chemometric application
class Food_Model(hdc.HD_Model):
    absorbance_start = None
    wavenum_start = None
    AM = None

    def __init__(self, D, encoding_scheme):
        hdc.HD_Model.__init__(self, D)
        self.encoding_scheme = encoding_scheme
        self.ppm_vals = [0, 2, 5, 10, 15] # Also add labels in lines: 288-289

    @staticmethod
    def _single_train(features): # Trains on a single row of the dataset
        print("Training on file: {}".format(features[1]))
        label = int(features[0])
        absorbances = list(map(float, features[2:]))
        result = schemes[sys.argv[3]](absorbances, Food_Model.absorbance_start, Food_Model.wavenum_start, D)
        return (label, result)

    def train(self): # Multithreaded to train on each sample in parallel
        dataset_length = len(self.trainset)
        print("Beginning training...")

        for label in self.ppm_vals:
            self.AM[label] = np.zeros(self.D)

        thread_pool = mp.Pool(mp.cpu_count())
        results = thread_pool.map(Food_Model._single_train, self.trainset)
        thread_pool.close()
        thread_pool.join()

        for result in results:
            label = result[0]
            encoding_scheme_result = result[1]
            self.AM[label] += encoding_scheme_result

        print("Trained on {} samples\n".format(dataset_length))

        for key in self.AM:
            self.AM[key] = binarizeHV(self.AM[key], 0)

        Food_Model.AM = self.AM

    @staticmethod
    def _single_test(features): # Tests on a single row of the dataset
        print("Testing on file: {}".format(features[1]))
        label = int(features[0])
        absorbances = list(map(float, features[2:]))
        result = schemes[sys.argv[3]](absorbances, Food_Model.absorbance_start, Food_Model.wavenum_start, D)
        query_hv = binarizeHV(result, 0)
        predicted = hdc.query(Food_Model.AM, query_hv)
        correct = int(predicted == label)
        f1 = None

        if predicted == label:
            if predicted == 0 or predicted == 2:
                f1 = "TN"
            else:
                f1 = "TP"
        else:
            if predicted == 0:
                if label == 2:
                    f1 = "TN"
                else:
                    f1 = "FN"
            elif predicted == 2:
                if label == 0:
                    f1 = "TN"
                else:
                    f1 = "FN"
            elif predicted == 5:
                if label == 0 or label == 2:
                    f1 = "FP"
                else:
                    f1 = "TP"
            elif predicted == 10:
                if label == 0 or label == 2:
                    f1 = "FP"
                else:
                    f1 = "TN"
            elif predicted == 15:
                if label == 0 or label == 2:
                    f1 = "FP"
                else:
                    f1 = "TP"

        return (correct, f1)

    def test(self): # Multithreaded to test on each sample in parallel
        print("Beginning testing...")
        dataset_length = len(self.testset)
        total = dataset_length
        correct = 0
        f1_params = {}
        f1_params["TN"] = 0
        f1_params["TP"] = 0
        f1_params["FN"] = 0
        f1_params["FP"] = 0

        thread_pool = mp.Pool(mp.cpu_count())
        results = thread_pool.map(Food_Model._single_test, self.testset)
        thread_pool.close()
        thread_pool.join()

        for result in results:
            correct += result[0]
            f1_result = result[1]
            f1_params[f1_result] += 1

        accuracy = correct / total
        f1 = 2 * f1_params["TP"] / (2 * f1_params["TP"] + f1_params["FP"] + f1_params["FN"])

        print("Tested on {} samples\n".format(dataset_length))
        print("Accuracy: {}%".format(round(accuracy * 100, 2)))
        print("F1 score: {}".format(round(f1, 2)))

        return accuracy, f1

    def _filter_dataset(self, dataset, name): # Retreives data for a specific category
        filtered_dataset = []

        for row in dataset:
            if name in row[1]:
                filtered_dataset.append(row)

        return np.array(filtered_dataset)

    def load_dataset(self, fraction_train):
        self.fraction_train = fraction_train
        file = open(sys.argv[1], "r")
        self.dataset = file.read().splitlines()[1:]

        for i in range(0, len(self.dataset)):
            self.dataset[i] = self.dataset[i].split(",")

        self.dataset = self._filter_dataset(self.dataset, category_names[sys.argv[2]])
        np.random.shuffle(self.dataset)
        self.split_train_and_test()

    def retrieve_indices_of_all_labels(self): # Finds all the rows for all labels of a specific category
        self.ppms = {}

        for label in self.ppm_vals:
            self.ppms[label] = self.retrieve_indices_of_label(label)

    def retrieve_indices_of_label(self, label): # Finds all the rows corresponding to a specific label
        indices = []

        for i in range(0, len(self.dataset)):
            if int(self.dataset[i][0]) == label:
                indices.append(i)

        return indices

    def split_train_and_test(self): # Splits the dataset into training and testing sets
        self.retrieve_indices_of_all_labels()
        num_files_per_category = int(sys.argv[4])
        ppm_indices = {}
        ppm_samples = {}

        for label in self.ppm_vals:
            ppm_indices[label] = self.ppms[label][0 : num_files_per_category]
            ppm_samples[label] = np.copy(self.dataset[ppm_indices[label]])

        self.trainset = np.concatenate((ppm_samples[0], ppm_samples[2], ppm_samples[5], ppm_samples[10], ppm_samples[15]))
        self.testset = np.delete(self.dataset, (ppm_indices[0] + ppm_indices[2] + ppm_indices[5] + ppm_indices[10] + ppm_indices[15]), axis=0)

def save(obj, file_name): # Serializes the model to a file
    file = open(file_name, "wb")
    pickle.dump(obj, file)
    file.close()

def load(file_name): # Deserializes the file to a model
    file = open(file_name, "rb")
    obj = pickle.load(file)
    file.close()
    return obj

def main():
    programStartTime = time.time()
    food_model = Food_Model(D, encoding_scheme=schemes[sys.argv[3]])
    food_model.load_dataset(float(sys.argv[4]))
    food_model.gen_iM(["wavenum_start"])
    food_model.gen_iM(["absorbance_start"])

    Food_Model.wavenum_start = food_model.iM["wavenum_start"]
    Food_Model.absorbance_start = food_model.iM["absorbance_start"]

    food_model.train()
    accuracy, f1 = food_model.test()

    programEndtTime = time.time()
    print("Runtime: {} seconds".format(round(programEndtTime - programStartTime, 2)))

    return accuracy, f1


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 hdc_model.py name_of_dataset category_num scheme num_files_per_category num_runs name_of_output_file ")
        print("e.g. python3 hdc_model.py datasets/our_aggregate_data.csv 1 m 2 10 output.txt")
        print("\nFor schemes: \nc: convolution\nm: multiplication\nt: trigram")
        print("\nCategories:- \n1: DNA ECOLI\n2: DNA Anodisc\n3: DNA In-Liquid DNA\n4: Yeast In-Liquid HK\n5: Yeast In-Liquid Live\n")

    else:
        NUM_RUNS = int(sys.argv[5])
        threshold = threshold_values[category_names[sys.argv[2]]]
        file = open(sys.argv[6], "w")
        accuracies = []
        f1s = []

        TotalRunStart = time.time();
        for i in range(0, NUM_RUNS):
            print("RUN {}".format(i))
            accuracy, f1 = main()
            accuracies.append(accuracy)
            f1s.append(f1)
            file.write("RUN: " + str(i) + " " + str(accuracy) + ", " + str(f1) + "\n")
        TotalRunEnd = time.time();

        avg_accuracy = stats.mean(accuracies)
        avg_f1 = stats.mean(f1s)
        std_accuracy = stats.stdev(accuracies, xbar=avg_accuracy)
        std_f1 = stats.stdev(f1s, xbar=avg_f1)

        file.write("Average Accuracy: " + str(avg_accuracy) + "\n")
        file.write("Std Accuracy: " + str(std_accuracy) + "\n")
        file.write("Average F1: " + str(avg_f1) + "\n")
        file.write("Std F1: " + str(std_f1) + "\n")

        print()
        print("Num Runs Done: {}".format(NUM_RUNS))
        print("Average Accuracy: {}% +- {}%".format(round(avg_accuracy * 100, 2), round(std_accuracy * 100, 2)))
        print("Average F1: {} +- {}".format(round(avg_f1, 2), round(std_f1, 2)))
        print("Total Runtime: {} seconds".format(round(TotalRunEnd - TotalRunStart, 2)))

        file.close()
