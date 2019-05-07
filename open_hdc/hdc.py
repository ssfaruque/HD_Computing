import numpy as np
import random as rand
import math
import pickle
from abc import ABC, abstractmethod


def gen_rand_hv(D):
    hv = np.empty(D)
    indices = rand.sample(range(D), D) # an array of random unique integers in the range [0, D)
    hv[indices[0 : int(D/2)]] = -1
    hv[indices[int(D/2) : D]] = +1

    return hv

def gen_rand_hv1(D):
    hv = np.empty(D)
    indices = rand.sample(range(D), D) # an array of random unique integers in the range [0, D)
    hv[indices[0 : int(D/5)]] = -1
    hv[indices[int(D/5) : D]] = +1

    return hv

def gen_rand_hv2(D):
    hv = np.empty(D)
    indices = rand.sample(range(D), D) # an array of random unique integers in the range [0, D)
    hv[indices[0 : int(2*D/5)]] = -1
    hv[indices[int(2*D/5) : D]] = +1

    return hv

def cos_angle(hv1, hv2):
    return sum((hv1 * hv2)) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))


def binarizeHV(hv, threshold):
    for i in range(len(hv)):
        if hv[i] > threshold:
            hv[i] = 1
        else:
            hv[i] = -1
    return hv


def gen_n_gram_sum(text, iM, D, n):
    start = 0
    end = n

    sum_hv = np.zeros(D)
    line = text.split("\n")[0]

    while end < len(line):
        letters = line[start:end]
        num_shifts = n - 1

        prod_hv = np.ones(D)

        for c in letters:
            letter_hv = np.roll(iM.get(c, np.zeros(D)), num_shifts)

            prod_hv *= letter_hv
            num_shifts -= 1

        sum_hv += prod_hv
        start += 1
        end += 1

    return sum_hv



class HD_Model(ABC):
    def __init__(self, D=10000):
        self.D = D
        self.iM  = {}
        self.AM  = {}
        self.dataset = []
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    def _cos_angle(self, hv1, hv2):
        return sum((hv1 * hv2)) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))
        

    def query(self, query_hv, print_all=False):
        label = None
        maxAngle = -1

        for key in self.AM:
            similarity = self._cos_angle(self.AM[key], query_hv)
            if  similarity > maxAngle:
                maxAngle = similarity
                label = key

            if print_all:
                print(key, ":", similarity)

        if print_all:
            print()

        return label

    def gen_iM(self, list_of_symbols, D):
        for symbol in list_of_symbols:
            self.iM[symbol] = gen_rand_hv(D)


    def gen_CiM(self, min_val, max_val, m):
        indices = rand.sample(range(self.D), self.D // 2)
        num_bits_to_flip = math.floor((self.D / 2) / (m - 1))
        numerical_step = (max_val - min_val) / (m - 1)
        curr_numerical_val = min_val
        hv = gen_rand_hv(self.D)

        start = 0
        end = num_bits_to_flip

        for i in range(m):
            self.iM[curr_numerical_val] = np.copy(hv)
            hv[indices[start : end]] *= -1
            start = end
            end += num_bits_to_flip
            curr_numerical_val += numerical_step



def save(obj, file_name):
    file = open(file_name, "wb")
    pickle.dump(obj, file)
    file.close()

def load(file_name):
    file = open(file_name, "rb")
    obj = pickle.load(file)
    file.close()
    return obj




def func(D):
    hv = []

    for i in range(1000):
        hv.append(gen_rand_hv(D))

    hv.append(gen_rand_hv1(D))
    hv.append(gen_rand_hv2(D))


    A = np.zeros(D)

    for i in range(1000):
        A = A + hv[i]


    B = binarizeHV(A + hv[1000], 0)
    C = binarizeHV(A + hv[1001], 0)

    similarity = cos_angle(B, C) 

    print(similarity)


def main():
    D = 10000

    for i in range(100):
        func(100)




    """
    1. Create HD_Model object
    2. Generate necessary item memories
    3. Load dataset and train on it
    4. Query the model / Use testing set
    5. Save model to file (optional)
    """


if __name__ == "__main__":
    main()