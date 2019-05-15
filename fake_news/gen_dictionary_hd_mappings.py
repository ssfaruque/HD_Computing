import numpy as np
import random as rand
import sys
import pickle


def save(obj, file_name):
    file = open(file_name, "wb")
    pickle.dump(obj, file)
    file.close()

def gen_rand_hv(D):
    hv = np.empty(D)
    indices = rand.sample(range(D), D) # an array of random unique integers in the range [0, D)
    hv[indices[0 : int(D/2)]] = -1
    hv[indices[int(D/2) : D]] = +1



def main():
	file_name = sys.argv[1]

	iM = {}

	with open(file_name, 'r') as file:
		for line in file:
			word = line.strip()
			iM[word] = gen_rand_hv(10000)

	save(iM, 'english_words_iM.pickle')









if __name__ == '__main__':
	main()