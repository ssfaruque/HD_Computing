import hdc
import sys
import random as rand
import numpy as np
import math
import pickle

D = 20000
rand_indices = rand.sample(range(D), D // 2)

# generate the iM corresponding to an absorbance on the fly
def calc_abs_iM(min_hv, abs_val, D, m):
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
            absorbance_hv = calc_abs_iM(min_abs_hv, absorbances[index], D, m=10001)
            wavenum_hv = calc_wn_iM(min_wn_hv, index, D, m=(len(absorbances) + 1))
            #tmp_hv = absorbance_hv * wavenum_hv
            tmp_hv = np.convolve(absorbance_hv, wavenum_hv, mode="same")
            #tmp_hv = np.roll(absorbance_hv, num_shifts)
            prod_hv *= tmp_hv

            num_shifts -= 1


    	#absorbance_hv = calc_abs_iM(min_abs_hv, absorbances[index], D, m=1001)
    	#wavenum_hv = calc_wn_iM(min_wn_hv, index, D, m=(len(absorbances) + 1))
    	sum_hv += prod_hv
    	index += 1
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



fraction = 0.85


class Food_Model(hdc.HD_Model):
	def __init__(self, D):
		hdc.HD_Model.__init__(self, D)

	def train(self):
		#wavenum_step = 1.928816
		dataset_length = len(self.dataset)
		end_mark = math.floor(fraction * dataset_length)

		print("Beginning training...")

		for i in range(0, end_mark):
			print("Training on file: {}".format(self.dataset[i][1]))
			label       = int(self.dataset[i][0])
			absorbances = self.dataset[i][2:]
			absorbances = list(map(float, absorbances))

			if label not in self.AM:
				self.AM[label] = np.zeros(self.D)

			ngram_sum = gen_n_gram_sum(absorbances, self.iM["absorbance_start"], self.iM["wavenum_start"], self.D, n=1)
			self.AM[label] += ngram_sum

			print("{}% complete".format( round((i+1)*100/end_mark,2) ))


		# binarize the AMs
		for key in self.AM:
			self.AM[key] = binarizeHV(self.AM[key], 0)




	def test(self):

		print("Beginning testing...")
		dataset_length = len(self.dataset)
		total = 0
		correct = 0

		beg_mark = math.floor(fraction * dataset_length)


		for i in range(beg_mark, dataset_length):
		    print("Testing on file:{}".format(self.dataset[i][1]))
		    label       = int(self.dataset[i][0])
		    absorbances = self.dataset[i][2:]
		    absorbances = list(map(float, absorbances))

		    ngram_sum = gen_n_gram_sum(absorbances, self.iM["absorbance_start"], self.iM["wavenum_start"], self.D, n=1)
		    query_hv = binarizeHV(ngram_sum, 0)
		    predicted = self.query(query_hv)

		    print("predicted: {}, ground truth: {}".format(predicted, label))

		    if predicted == label:
		        correct += 1

		    total += 1

		accuracy = correct / total
		f1 = (2 * correct) / (correct + total)

		print("accuracy: {}".format(accuracy))
		print("f1 score: {}".format(f1))

		#file = open("out0_15.txt", "a")
		#file.write(str(accuracy) + "\n")
		#file.close()



	def load_dataset(self):
		file = open(sys.argv[1], "r")
		self.dataset = file.read().splitlines()
		self.dataset = self.dataset[1:]
		for i in range(0, len(self.dataset)):
			self.dataset[i] = self.dataset[i].split(",")

		#rand.shuffle(self.dataset)

		self.dataset = filter_dataset(self.dataset, "inliquidHK")
		rand.shuffle(self.dataset)


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
	food_model.gen_iM(["wavenum_start"], D)
	food_model.gen_iM(["absorbance_start"], D)
	#food_model.iM["absorbance_end"] = gen_max_hv(food_model.iM["absorbance_start"], D)


	food_model.train()
	save(food_model, "model.bin")
	food_model.test()





if __name__ == "__main__":
	main()
