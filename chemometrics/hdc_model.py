import hdc
import sys
import random as rand
import numpy as np
import math
import pickle
import time

from sklearn.preprocessing import StandardScaler

D = 10000
rand_indices = rand.sample(range(D), D // 2)

# generate the iM corresponding to an absorbance on the fly
threshold = 0.07
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

def gen_n_gram_sum(absorbances, min_abs_hv, min_wn_hv, D, n):
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
            #tmp_hv = np.convolve(absorbance_hv, wavenum_hv, mode="same")
            #tmp_hv = np.roll(absorbance_hv, num_shifts)
            prod_hv *= tmp_hv
            index += 1
            num_shifts -= 1


        sum_hv += prod_hv
        index -= (n - 1)
        start += 1
        end += 1

    return sum_hv

def binarizeHV(hv, threshold):
    for i in range(len(hv)):
        if hv[i] > threshold:
            hv[i] = 1
        else:
            hv[i] = -1
    return hv

def gen_max_hv(start_hv, D):
    hv = np.copy(start_hv)
    hv[rand_indices[0 : int(D / 2)]] *= -1
    return hv

def filter_dataset(dataset, name):
    filtered_dataset = []

    for row in dataset:
        if name in row[1]:
            filtered_dataset.append(row)
    return filtered_dataset



def find_labels(dataset, label):
    return [row for index, row in enumerate(dataset) if int(row[0]) == label]


class Food_Model(hdc.HD_Model):
    def __init__(self, D):
        hdc.HD_Model.__init__(self, D)

    def train(self):
        #wavenum_step = 1.928816
        dataset_length = len(self.trainset)

        print("Beginning training...")

        for i in range(0, dataset_length):
            print("Training on file: {}".format(self.trainset[i][1]))
            label       = int(self.trainset[i][0])
            absorbances = self.trainset[i][2:]
            absorbances = list(map(float, absorbances))

            if label not in self.AM:
                self.AM[label] = np.zeros(self.D)

            ngram_sum = gen_n_gram_sum(absorbances, self.iM["absorbance_start"], self.iM["wavenum_start"], self.D, n=1)
            self.AM[label] += ngram_sum

            print("{}% complete".format( round((i + 1) * 100 / dataset_length, 2) ))

        print("Trained on {} samples\n".format(dataset_length))

        # binarize the AMs
        for key in self.AM:
            self.AM[key] = binarizeHV(self.AM[key], 0)

    def test(self):

        print("Beginning testing...")
        dataset_length = len(self.testset)
        total = 0
        correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        self.correct_count = {}
        self.correct_count[0] = 0
        self.correct_count[2] = 0
        self.correct_count[5] = 0
        self.correct_count[10] = 0
        self.correct_count[15] = 0


        for i in range(0, dataset_length):
            print("Testing on file:{}".format(self.testset[i][1]))
            label       = int(self.testset[i][0])
            absorbances = self.testset[i][2:]
            absorbances = list(map(float, absorbances))

            ngram_sum = gen_n_gram_sum(absorbances, self.iM["absorbance_start"], self.iM["wavenum_start"], self.D, n=1)
            query_hv = binarizeHV(ngram_sum, 0)
            predicted = self.query(query_hv)

            print("{}% complete\t Guess: {}\t Truth: {}".format(round((i + 1) * 100 / dataset_length, 2), predicted, label))

            if predicted == label:
                correct += 1
                self.correct_count[predicted] += 1
                if predicted == 0 or predicted == 2:
                    TN += 1
                else:
                    TP += 1
            else:
                if predicted == 0:
                    if label == 2:
                        TN += 1
                    else:
                        FN += 1
                elif predicted == 2:
                    if label == 0:
                        TN += 1
                    else:
                        FN += 1
                elif predicted == 5:
                    if label == 0 or label == 2:
                        FP += 1
                    else:
                        TP += 1
                elif predicted == 10:
                    if label == 0 or label == 2:
                        FP += 1
                    else:
                        TN += 1
                elif predicted == 15:
                    if label == 0 or label == 2:
                        FP += 1
                    else:
                        TP += 1

            total += 1


        print("Tested on {} samples\n".format(dataset_length))
        accuracy = correct / total
        #precision = TP / (TP + FP)
        #recall = TP / (TP + FN)
        #f1 = 2 * precision * recall / (precision + recall)
        f1 = 2 * TP / (2 * TP + FP + FN)

        print("Accuracy: {}%".format(round(accuracy * 100, 2)))
        print("F1 score: {}".format(round(f1, 2)))
        #file = open("out0_15.txt", "a")
        #file.write(str(accuracy) + "\n")
        #file.close()

        return accuracy, f1

    def _standardize_dataset(self, dataset):
        standardized_dataset = np.array(dataset)
        meta_data = standardized_dataset[:, 0:2]
        standardized_dataset = standardized_dataset[:, 2:]
        standardized_dataset = standardized_dataset.astype(float)

        scaler = StandardScaler()
        scaler.fit(standardized_dataset)
        standardized_dataset = scaler.transform(standardized_dataset)

        return np.concatenate((meta_data, standardized_dataset), axis=1)


    def load_dataset(self, fraction_train):
        self.fraction_train = fraction_train
        file = open(sys.argv[1], "r")
        self.dataset = file.read().splitlines()
        self.dataset = self.dataset[1:]
        for i in range(0, len(self.dataset)):
            self.dataset[i] = self.dataset[i].split(",")


        self.dataset = filter_dataset(self.dataset, "Yeast_inliquid Live")
        rand.shuffle(self.dataset)

        #self.dataset = self._standardize_dataset(self.dataset)

        split_mark = math.floor(self.fraction_train * len(self.dataset))
        self.trainset = self.dataset[0 : split_mark]
        self.testset = self.dataset[split_mark :]






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
    #D = 10000
    programStartTime = time.time()
    food_model = Food_Model(D)
    food_model.load_dataset(float(sys.argv[2]))
    food_model.gen_iM(["wavenum_start"], D)
    food_model.gen_iM(["absorbance_start"], D)
    #food_model.iM["absorbance_end"] = gen_max_hv(food_model.iM["absorbance_start"], D)


    food_model.train()
    #save(food_model, "model.bin")
    accuracy = food_model.test() #accuracy holds accuracy and F1 both
    programEndtTime = time.time()
    print("Runtime: {} seconds".format(round(programEndtTime - programStartTime, 2)))

    return accuracy #accuracy holds accuracy and F1 both






if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 hdc_model.py name_of_dataset fraction_train num_runs name_of_output_file ")
        print("e.g. python3 hdc_model.py datasets/our_aggregate_data.csv 0.7 10 output.txt")

    else:
        NUM_RUNS = int(sys.argv[3])
        file = open(sys.argv[4], "w")
        accuracies = []
        f1s = []

        TotalRunStart = time.time();
        for i in range(0, NUM_RUNS):
            print("RUN {}".format(i))
            accuracy, f1 = main()
            accuracies.append(accuracy)
            f1s.append(f1)
            file.write(str(accuracy) + "\n")
            file.write(str(f1) + "\n")
        TotalRunEnd = time.time();

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_f1 = sum(f1s) / len(f1s)
        file.write("Average Accuracy: " + str(avg_accuracy) + "\n")
        file.write("Average F1: " + str(avg_f1) + "\n")
        print("Num Runs Done: {}".format(NUM_RUNS))
        print("Average Accuracy: {}%".format(round(avg_accuracy * 100, 2)))
        print("Average F1: {}".format(round(avg_f1, 2)))
        print("Total Runtime: {} seconds".format(round(TotalRunEnd - TotalRunStart, 2)))

        file.close()
