
import sys
import random as rand
import math
import numpy as np

from sklearn import datasets
from sklearn import svm


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
		end_mark = math.floor(self.fraction_split * len(self.dataset))
		self.model.fit(self.dataset[0:end_mark, 2:], self.dataset[0:end_mark, 0])

	def test(self):
		beg_mark = math.floor(self.fraction_split * len(self.dataset))
		total = 0
		correct = 0

		for i in range(beg_mark, len(self.dataset)):
			label = int(self.dataset[i][0])
			print("Label: " + str(label), end = ", ")

			prediction = int(self.model.predict([self.dataset[i, 2:]]))
			print(" Prediction: " + str(prediction))

			if label == prediction:
				correct += 1

			total += 1

		accuracy = correct / total

		print("Accuracy:", accuracy)



	def load_dataset(self):
		file = open(sys.argv[1], "r")
		self.dataset = file.read().splitlines()
		self.dataset = self.dataset[1:]
		for i in range(0, len(self.dataset)):
			self.dataset[i] = self.dataset[i].split(",")

		self.dataset = filter_dataset(self.dataset, "Yeast_inliquid HK")
		rand.shuffle(self.dataset)
		self.dataset = np.array(self.dataset)



def main():

	svm = SVM(C=100000, fraction_split=0.7)
	svm.load_dataset()
	svm.train()
	svm.test()




if __name__ == "__main__":
	main()