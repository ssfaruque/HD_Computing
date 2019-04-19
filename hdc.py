import numpy as np
import random as rand
import math


def gen_rand_hv(D):
    hv = np.empty(D)
    indices = rand.sample(range(D), D) # an array of random unique integers in the range [0, D)
    hv[indices[0 : int(D/2)]] = -1
    hv[indices[int(D/2) : D]] = +1

    return hv


def gen_item_mem(list_of_symbols, D):
    item_mem = {}

    for symbol in list_of_symbols:
        item_mem[symbol] = gen_rand_hv(D)

    return item_mem


"""
def gen_cont_item_mem(min_val, max_val, D, m):
    cont_item_mem = {}
    indices = rand.sample(range(D), D // 2)

    num_bits_to_flip = math.floor((D / 2) / (m - 1))
    numerical_step = (max_val - min_val) / (m - 1)
    curr_numerical_val = min_val
    hv = gen_rand_hv(D)

    print(num_bits_to_flip)
    print(hv)

    start = 0
    end = num_bits_to_flip

    for i in range(m):
        print("start:{}, end:{}".format(start, end))
        cont_item_mem[curr_numerical_val] = hv
        hv[indices[start : end]] *= -1
        start = end
        end += num_bits_to_flip
        curr_numerical_val += numerical_step

    return cont_item_mem
"""

def cos_angle(hv1, hv2):
    return sum((hv1 * hv2)) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))



def mag(x):
    return math.sqrt(sum(i**2 for i in x))


def binarizeHV(hv, threshold):
    for i in range(len(hv)):
        if hv[i] > threshold:
            hv[i] = 1
        else:
            hv[i] = -1
    return hv



def learn_languages(n):
    D = 10000
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u' ,'v', 'w', 'x', 'y', 'z', ' ']
    lang_labels = ['afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav', 'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe']

    iM = gen_item_mem(alphabet + lang_labels, D)
    AM = {}


    for label in lang_labels:
        file = open("language_recognition_data/training_texts/" + label + ".txt", "r")
        sum_hv = np.zeros(D)

        for line in file:
            sum_hv += gen_n_gram_sum(line, iM, D, n)

        sum_hv = binarizeHV(sum_hv, 0)
        AM[label] = sum_hv

    return iM, AM


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
            letter_hv = np.roll(iM[c], num_shifts)
            prod_hv *= letter_hv
            num_shifts -= 1

        sum_hv += prod_hv
        start += 1
        end += 1

    return sum_hv



def find_label(query_hv, AM):
    label = "NONE"
    maxAngle = -1

    for key in AM:
        similarity = cos_angle(AM[key], query_hv)
        if  similarity > maxAngle:
            maxAngle = similarity
            label = key

        print(key, ": ", similarity)

    return label



   

def main():
    D = 10000
    n = 3

    iM, AM = learn_languages(3)

    test_text = "these are not serious things"

    query_hv = binarizeHV(gen_n_gram_sum(test_text, iM, D, n), 0)

    print(find_label(query_hv, AM))




if __name__ == "__main__":
    main()