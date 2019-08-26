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


D = 10000
threshold = 0
rand_indices = rand.sample(range(D), D // 2)


# dataset_name = sys.argv[1]
# category = sys.argv[2]
# scheme = sys.argv[3]
# num_splits = int(sys.argv[4])
# num_runs = int(sys.argv[5])
# output_file = sys.argv[6]


# generate the iM corresponding to an absorbance on the fly
def calc_abs_iM(min_hv, abs_val, D, m):
    if  abs_val < threshold:
        return np.zeros(D)
    num_bits_to_flip = math.floor((D / 2) / (m - 1))
    hv = np.copy(min_hv)

    step_size = 1 / (m - 1)
    level = round(abs_val / step_size)

    hv[rand_indices[0 : (num_bits_to_flip * level)]] *= -1

    return hv


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


def convolution(absorbances, min_abs_hv, min_wn_hv, D, n=1):
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
            wavenum_hv = calc_wn_iM(min_wn_hv, index, D, m=(len(absorbances) + 1))
            tmp_hv = np.convolve(absorbance_hv, wavenum_hv, mode="same")
            prod_hv *= tmp_hv
            index += 1
            num_shifts -= 1


        sum_hv += prod_hv
        index -= (n - 1)
        start += 1
        end += 1

    return sum_hv


def multiplication(absorbances, min_abs_hv, min_wn_hv, D, n=1):
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
            wavenum_hv = calc_wn_iM(min_wn_hv, index, D, m=(len(absorbances) + 1))
            tmp_hv = absorbance_hv * wavenum_hv
            prod_hv *= tmp_hv
            index += 1
            num_shifts -= 1


        sum_hv += prod_hv
        index -= (n - 1)
        start += 1
        end += 1

    return sum_hv

# ============================================================


schemes = {"convolution": convolution,
           "multiplication": multiplication,
           "trigram": trigram}

threshold_values = {"DNA_ECOLI": 0.065,
                    "Yeast_inliquid HK": 0.055,
                    "DNA_INLIQUIDDNA": 0.0875,
                    "DNA_DNA@Anod": 0.07,
                    "Yeast_inliquid Live": 0.07}


def binarizeHV(hv, threshold):
    for i in range(len(hv)):
        if hv[i] > threshold:
            hv[i] = 1
        else:
            hv[i] = -1
    return hv


class Food_Model(hdc.HD_Model):
    absorbance_start = None
    wavenum_start = None
    AM = None


    def __init__(self, D, encoding_scheme):
        hdc.HD_Model.__init__(self, D)
        self.encoding_scheme = encoding_scheme


    @staticmethod
    def _single_train(features):
        print("Training on file: {}".format(features[1]))
        label = int(features[0])
        absorbances = features[2:]
        absorbances = list(map(float, absorbances))
        result = schemes[sys.argv[3]](absorbances, Food_Model.absorbance_start, Food_Model.wavenum_start, D)
        return (label, result)

    def train(self):
        dataset_length = len(self.trainset)
        print("Beginning training...")

        self.AM[0] = np.zeros(self.D)
        self.AM[2] = np.zeros(self.D)
        self.AM[5] = np.zeros(self.D)
        self.AM[10] = np.zeros(self.D)
        self.AM[15] = np.zeros(self.D)

        thread_pool = mp.Pool(mp.cpu_count())
        results = thread_pool.map(Food_Model._single_train, self.trainset)
        thread_pool.close()
        thread_pool.join()

        for result in results:
            label = result[0]
            encoding_scheme_result = result[1]
            self.AM[label] += encoding_scheme_result

        """
        for i in range(0, dataset_length):
            print("Training on file: {}".format(self.trainset[i][1]))
            label       = int(self.trainset[i][0])
            absorbances = self.trainset[i][2:]
            absorbances = list(map(float, absorbances))

            #if label not in self.AM:
            #    self.AM[label] = np.zeros(self.D)

            result = self.encoding_scheme(absorbances, self.iM["absorbance_start"], self.iM["wavenum_start"], self.D)
            self.AM[label] += result

            print("{}% complete".format( round((i + 1) * 100 / dataset_length, 2) ))
        """

        print("Trained on {} samples\n".format(dataset_length))

        for key in self.AM:
            self.AM[key] = binarizeHV(self.AM[key], 0)

        Food_Model.AM = self.AM


    @staticmethod
    def _single_test(features):
        print("Testing on file: {}".format(features[1]))
        label = int(features[0])
        absorbances = features[2:]
        absorbances = list(map(float, absorbances))
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


    def test(self):
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

        """
        for i in range(0, dataset_length):
            print("Testing on file:{}".format(self.testset[i][1]))
            label       = int(self.testset[i][0])
            absorbances = self.testset[i][2:]
            absorbances = list(map(float, absorbances))

            ngram_sum = self.encoding_scheme(absorbances, self.iM["absorbance_start"], self.iM["wavenum_start"], self.D)
            query_hv = binarizeHV(ngram_sum, 0)
            predicted = self.query(query_hv)

            print("{}% complete\t Guess: {}\t Truth: {}".format(round((i + 1) * 100 / dataset_length, 2), predicted, label))

            if predicted == label:
                correct += 1

            f1_params = self._update_f1_results(f1_params, predicted, label)
            total += 1
        """

        print("Tested on {} samples\n".format(dataset_length))
        accuracy = correct / total
        f1 = 2 * f1_params["TP"] / (2 * f1_params["TP"] + f1_params["FP"] + f1_params["FN"])

        print("Accuracy: {}%".format(round(accuracy * 100, 2)))
        print("F1 score: {}".format(round(f1, 2)))

        return accuracy, f1


    def _filter_dataset(self, dataset, name):
        filtered_dataset = []

        for row in dataset:
            if name in row[1]:
                filtered_dataset.append(row)
        return np.array(filtered_dataset)


    def load_dataset(self, fraction_train):
        self.fraction_train = fraction_train
        file = open(sys.argv[1], "r")
        self.dataset = file.read().splitlines()
        self.dataset = self.dataset[1:]
        for i in range(0, len(self.dataset)):
            self.dataset[i] = self.dataset[i].split(",")

        self.dataset = self._filter_dataset(self.dataset, sys.argv[2])
        rand.shuffle(self.dataset)

        #split_mark = math.floor(self.fraction_train * len(self.dataset))
        #self.trainset = self.dataset[0 : split_mark]
        #self.testset = self.dataset[split_mark :]

    def update_train_and_test_sets(self, training_indices, testing_indices):
        self.trainset = self.dataset[training_indices]
        self.testset = self.dataset[testing_indices]


def save(obj, file_name):
    file = open(file_name, "wb")
    pickle.dump(obj, file)
    file.close()

def load(file_name):
    file = open(file_name, "rb")
    obj = pickle.load(file)
    file.close()
    return obj


def main():
    programStartTime = time.time()
    food_model = Food_Model(D, encoding_scheme=schemes[sys.argv[3]])
    food_model.load_dataset(float(sys.argv[4]))
    food_model.gen_iM(["wavenum_start"], D)
    food_model.gen_iM(["absorbance_start"], D)

    Food_Model.wavenum_start = food_model.iM["wavenum_start"]
    Food_Model.absorbance_start = food_model.iM["absorbance_start"]

    accuracies = []
    f1s = []
    num_splits = int(sys.argv[4])

    kf = KFold(n_splits=num_splits)
    split_num = 1

    for training_indices, testing_indices in kf.split(food_model.dataset):
        print("Split {}/{}".format(split_num, num_splits))
        food_model.update_train_and_test_sets(training_indices, testing_indices)
        food_model.train()
        accuracy, f1 = food_model.test()

        accuracies.append(accuracy)
        f1s.append(f1)

        split_num += 1

    programEndtTime = time.time()
    print("Runtime: {} seconds".format(round(programEndtTime - programStartTime, 2)))

    return stats.mean(accuracies), stats.mean(f1s)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 hdc_model.py name_of_dataset category scheme fraction_train num_runs name_of_output_file ")
        print("e.g. python3 hdc_model.py datasets/our_aggregate_data.csv ECOLI multiplication 0.7 10 output.txt")

    else:
        NUM_RUNS = int(sys.argv[5])
        threshold = threshold_values[sys.argv[2]]
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
