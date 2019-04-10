import numpy as np
import random as rand



import pprint


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


def gen_cont_item_mem(min_val, max_val, D, m):
    cont_item_mem = {}
    indices = rand.sample(range(D), D)

    step = int(D / 2 / (max_val - min_val) / m) + 1
    hv   = gen_rand_hv(D)

    #print(step)

    for i in range(m):
        cont_item_mem[i] = hv
        start = i * step
        end   = (i + 1) * step
        #print(hv)
        hv[indices[start : end]] *= -1

    return cont_item_mem


def cos_angle(hv1, hv2):
    return sum((hv1 * hv2)) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))


def main():
    cim = gen_cont_item_mem(0, 10, 100, 10)

    hv1 = gen_rand_hv(10000)
    hv2 = gen_rand_hv(10000)

    similarity = cos_angle(hv1, hv2)

    print("cosA:{}".format(similarity))




if __name__ == "__main__":
    main()