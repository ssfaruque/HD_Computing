
import sys
import random as rand
import math
import numpy as np
import statistics as stats

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold


threshold_values = {"DNA_ECOLI": 0.065,
                    "Yeast_inliquid HK": 0.055,
                    "DNA_INLIQUIDDNA": 0.0875,
                    "DNA_DNA@Anod": 0.07,
                    "Yeast_inliquid Live": 0.07}



def filter_dataset(dataset, name):
    filtered_dataset = []

    for row in dataset:
        if name in row[1]:
            filtered_dataset.append(row)
    return filtered_dataset


class SVM:
	def __init__(self, C=1.0, fraction_split=0.7):
		self.gamma = 'auto'
		self.C = C
		self.fraction_split = fraction_split # what fractino of dataset will be for training, remaining fraction for testing
		self.dataset = []
		self.model = svm.SVC(gamma=self.gamma, C=self.C)

	def train(self):
		self.model.fit(self.trainset[:, 2:], self.trainset[:, 0])
		print("Trained on " + str(len(self.trainset)) + " samples")

	def test(self):
		total = 0
		correct = 0
		f1_results = {}
		f1_results["TN"] = 0
		f1_results["TP"] = 0
		f1_results["FN"] = 0
		f1_results["FP"] = 0

		for i in range(0, len(self.testset)):
			label = int(self.testset[i][0])
			print("Label: " + str(label), end = ", ")

			predicted = int(self.model.predict([self.testset[i, 2:]]))
			print(" Prediction: " + str(predicted))


			if predicted == label:
				correct += 1
				if predicted == 0 or predicted == 2:
					f1_results["TN"] += 1
				else:
					f1_results["TP"] += 1
			else:
				if predicted == 0:
					if label == 2:
						f1_results["TN"] += 1
					else:
						f1_results["FN"] += 1
				elif predicted == 2:
					if label == 0:
						f1_results["TN"] += 1
					else:
						f1_results["FN"] += 1
				elif predicted == 5:
					if label == 0 or label == 2:
						f1_results["FP"] += 1
					else:
						f1_results["TP"] += 1
				elif predicted == 10:
					if label == 0 or label == 2:
						f1_results["FP"] += 1
					else:
						f1_results["TN"] += 1
				elif predicted == 15:
					if label == 0 or label == 2:
						f1_results["FP"] += 1
					else:
						f1_results["TP"] += 1


			total += 1

		accuracy = correct / total
		f1 = 2 * f1_results["TP"] / (2 * f1_results["TP"] + f1_results["FP"] + f1_results["FN"])

		print("Tested on " + str(len(self.testset)) + " samples")
		print("Accuracy:", accuracy)
		print("F1 Score", f1)

		return accuracy, f1


	def load_dataset(self, fraction_train):
		file = open(sys.argv[1], "r")
		self.dataset = file.read().splitlines()
		self.dataset = self.dataset[1:]
		for i in range(0, len(self.dataset)):
			self.dataset[i] = self.dataset[i].split(",")

		category = sys.argv[3]
		self.dataset = filter_dataset(self.dataset, category)
		np.random.shuffle(self.dataset)
		self.dataset = np.array(self.dataset)
		self.threshold_dataset()
		self.split_train_and_test()


	def threshold_dataset(self):
		threshold = threshold_values[sys.argv[3]]

		for i in range(len(self.dataset)):
			for j in range(2, len(self.dataset[0])):
				if float(self.dataset[i][j]) < threshold:
					self.dataset[i][j] = 0


	def retrieve_indices_of_label(self, label):
		indices = []
		for i in range(0, len(self.dataset)):
			if int(self.dataset[i][0]) == label:
				indices.append(i)
				return indices

	def retrieve_indices_of_all_labels(self):
		self.ppm0 = self.retrieve_indices_of_label(0)
		self.ppm2 = self.retrieve_indices_of_label(2)
		self.ppm5 = self.retrieve_indices_of_label(5)
		self.ppm10 = self.retrieve_indices_of_label(10)
		self.ppm15 = self.retrieve_indices_of_label(15)

	def split_train_and_test(self):
		self.retrieve_indices_of_all_labels()
		num_files_per_category = int(sys.argv[2])

		ppm0_indices = self.ppm0[0 : num_files_per_category]
		ppm2_indices = self.ppm2[0 : num_files_per_category]
		ppm5_indices = self.ppm5[0 : num_files_per_category]
		ppm10_indices = self.ppm10[0 : num_files_per_category]
		ppm15_indices = self.ppm15[0 : num_files_per_category]

		ppm0_samples = np.copy(self.dataset[ppm0_indices])
		ppm2_samples = np.copy(self.dataset[ppm2_indices])
		ppm5_samples = np.copy(self.dataset[ppm5_indices])
		ppm10_samples = np.copy(self.dataset[ppm10_indices])
		ppm15_samples = np.copy(self.dataset[ppm15_indices])

		self.trainset = np.concatenate((ppm0_samples, ppm2_samples, ppm5_samples, ppm10_samples, ppm15_samples))
		self.testset = np.delete(self.dataset, (ppm0_indices + ppm2_indices + ppm5_indices + ppm10_indices + ppm15_indices), axis=0)



def main():
	svm = SVM(C=100000, fraction_split=0.7)

	fraction_train = float(sys.argv[2])
	svm.load_dataset(fraction_train)
	svm.train()
	accuracy, f1 = svm.test()

	return accuracy, f1



if __name__ == "__main__":

	if len(sys.argv) != 5:
		print("Usage: python3 svm_model.py dataset num_files_per_category category output_file")
		exit()

	NUM_RUNS = 10
	file = open(sys.argv[4], "w")
	accuracies = []
	f1s = []

	for i in range(0, NUM_RUNS):
		print("RUN {}".format(i))
		accuracy, f1 = main()
		accuracies.append(accuracy)
		f1s.append(f1)
		file.write("Accuracy: " + str(accuracy) + ", "  + "f1: " + str(f1) + "\n")


	avg_accuracy = stats.mean(accuracies)
	avg_f1 = stats.mean(f1s)
	std_accuracy = stats.stdev(accuracies, xbar=avg_accuracy)
	std_f1 = stats.stdev(f1s, xbar=avg_f1)

	file.write("avg_accuracy: " + str(avg_accuracy) + " +- " + str(std_accuracy) + "\n")
	file.write("avg_f1: " + str(avg_f1) + " +- "  + str(std_f1) + "\n")

	print("Average Accuracy: {}% +- {}%".format(round(avg_accuracy * 100, 2), round(std_accuracy * 100, 2)))
	print("Average F1: {} +- {}".format(round(avg_f1, 2), round(std_f1, 2)))

	file.close()
