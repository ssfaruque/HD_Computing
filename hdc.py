import numpy as np
import random as rand
import math


def gen_rand_hv(D):
    hv = np.empty(D)
    indices = rand.sample(range(D), D) # an array of random unique indices in the range [0, D)
    hv[indices[0 : int(D/2)]] = -1
    hv[indices[int(D/2) : D]] = +1

    return hv


def gen_item_mem(list_of_symbols, D):
    item_mem = {}

    for symbol in list_of_symbols:
        item_mem[symbol] = gen_rand_hv(D)

    return item_mem



# Buggy, producing same hv for each numerical value
def gen_cont_item_mem(min_val, max_val, D, m):
    cont_item_mem = {}
    indices = rand.sample(range(D), D)

    num_bits_to_flip = math.floor((D / 2) / (m - 1))
    numerical_step = (max_val - min_val) / (m - 1)
    curr_numerical_val = min_val
    hv = gen_rand_hv(D)

    print(num_bits_to_flip)
    print(hv)

    start = 0
    end = num_bits_to_flip

    for i in range(m):
        cont_item_mem[curr_numerical_val] = hv
        hv[indices[start : end]] *= -1
        start = end
        end += num_bits_to_flip
        curr_numerical_val += numerical_step

    return cont_item_mem


def cos_angle(hv1, hv2):
    return sum((hv1 * hv2)) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))



def mag(x):
    return math.sqrt(sum(i**2 for i in x))

def main():
    print("norm result:{}".format(np.linalg.norm(hv)))
    print("magnitude result:{}".format(mag(hv)))




if __name__ == "__main__":
    main()