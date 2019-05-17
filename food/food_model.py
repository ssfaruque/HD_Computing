import hdc
import sys
import random as rand
import numpy as np
import math

D = 10000
rand_indices = rand.sample(range(D), D // 2)


def calc_abs_iM(rand_indices, min_hv, index, D, m):
    num_bits_to_flip = math.floor((D / 2) / (m - 1))
    hv = min_hv

    start = 0
    end = num_bits_to_flip

    for i in range(index):
        hv[rand_indices[start : end]] *= -1
        start = end
        end += num_bits_to_flip

    return hv




def gen_n_gram_sum(absorbances, min_hv, D, n):
    start = 0
    end = n
    index = 0

    sum_hv = np.zeros(D)

    while end < len(absorbances) - n + 1:
        tri_absorbance = absorbances[start:end]
        num_shifts = n - 1

        prod_hv = np.ones(D)

        for absorbance in tri_absorbance:
        	tmp_hv = calc_abs_iM(rand_indices, min_hv, index, D, 3)
        	absorbance_hv = np.roll(tmp_hv, num_shifts)
        	prod_hv *= absorbance_hv
        	num_shifts -= 1
        	index += 1

        index -= 2
        sum_hv += prod_hv
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
    indices = rand.sample(range(D), D // 2)
    hv = np.copy(start_hv)
    hv[indices[0 : int(D / 2)]] *= -1
    return hv


class Food_Model(hdc.HD_Model):
	def __init__(self, D=10000):
		hdc.HD_Model.__init__(self, D)

	def train(self):
		#wavenum_step = 1.928816
		dataset_length = 101

		print("Beginning training...")

		for i in range(1, dataset_length):
			print("Training on file:{}".format(self.dataset[i][1]))
			label       = int(self.dataset[i][0])
			absorbances = self.dataset[i][2:]
			absorbances = list(map(float, absorbances))

			if label not in self.AM:
				self.AM[label] = np.zeros(self.D)

			ngram_sum = gen_n_gram_sum(absorbances, self.iM["absorbance_start"], self.D, n=3)
			self.AM[label] += ngram_sum

		# binarize the AMs
		for key in self.AM:
			self.AM[key] = binarizeHV(self.AM[key], 0)







	def test(self):
		pass

	def load_dataset(self):
		file = open(sys.argv[1], "r")
		self.dataset = file.read().splitlines()
		for i in range(0, len(self.dataset)):
			self.dataset[i] = self.dataset[i].split(",")

		#rand.shuffle(self.dataset)


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
	food_model = Food_Model(D)
	food_model.load_dataset()
	food_model.gen_iM(["wavenum_start"], D=10000)
	food_model.gen_iM(["absorbance_start"], D=10000)
	food_model.iM["absorbance_end"] = gen_max_hv(food_model.iM["absorbance_start"], D)
	food_model.train()

	save(food_model, "model.bin")

	print(food_model.AM)



if __name__ == "__main__":
	main()