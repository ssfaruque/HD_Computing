
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

	def test(self):
		beg_mark = math.floor(self.fraction_split * len(self.dataset))
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

		print("Accuracy:", accuracy)
		print("F1 Score", f1)

		return accuracy, f1


	def load_dataset(self):
		file = open(sys.argv[1], "r")
		self.dataset = file.read().splitlines()
		self.dataset = self.dataset[1:]
		for i in range(0, len(self.dataset)):
			self.dataset[i] = self.dataset[i].split(",")

		category = sys.argv[3]
		self.dataset = filter_dataset(self.dataset, category)
		rand.shuffle(self.dataset)
		self.dataset = np.array(self.dataset)
		self.threshold_dataset()


	def threshold_dataset(self):
		threshold = threshold_values[sys.argv[3]]

		for i in range(len(self.dataset)):
			for j in range(2, len(self.dataset[0])):
				if float(self.dataset[i][j]) < threshold:
					self.dataset[i][j] = 0





	def update_train_and_test_sets(self, training_indices, testing_indices):
		self.trainset = self.dataset[training_indices]
		self.testset = self.dataset[testing_indices]


def main():
	svm = SVM(C=100000, fraction_split=0.7)
	svm.load_dataset()

	accuracies = []
	f1s = []
	num_splits = int(sys.argv[2])

	kf = KFold(n_splits=num_splits)
	split_num = 1


	for training_indices, testing_indices in kf.split(svm.dataset):
		print("Split {}/{}".format(split_num, num_splits))
		svm.update_train_and_test_sets(training_indices, testing_indices)
		svm.train()
		accuracy, f1 = svm.test()

		accuracies.append(accuracy)
		f1s.append(f1)

		split_num += 1

	return stats.mean(accuracies), stats.mean(f1s)




if __name__ == "__main__":

	if len(sys.argv) != 5:
		print("Usage: python3 svm_model.py dataset num_splits category output_file")
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
