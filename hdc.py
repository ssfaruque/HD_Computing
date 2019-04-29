import numpy as np
import random as rand
import math
import pickle
import os

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



def gen_cont_item_mem(min_val, max_val, D, m):
    cont_item_mem = {}
    indices = rand.sample(range(D), D // 2)

    num_bits_to_flip = math.floor((D / 2) / (m - 1))
    numerical_step = (max_val - min_val) / (m - 1)
    curr_numerical_val = min_val
    hv = gen_rand_hv(D)

    print(indices)
    print("num_bits_to_flip", num_bits_to_flip)
    print("numerical_step", numerical_step)
    print("hv", hv)

    start = 0
    end = num_bits_to_flip

    for i in range(m):
        print("start:{}, end:{}".format(start, end))
        cont_item_mem[curr_numerical_val] = np.copy(hv)
        print("indices used: ", indices[start : end])
        #print(curr_numerical_val, hv)
        hv[indices[start : end]] *= -1
        start = end
        end += num_bits_to_flip
        curr_numerical_val += numerical_step

    return cont_item_mem


def cos_angle(hv1, hv2):
    return sum((hv1 * hv2)) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))


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

        print("Starting " + str(n) + "-gram sum computing for: " + (label + ".txt") + "...")

        for line in file:
            sum_hv += gen_n_gram_sum(line, iM, D, n)

        sum_hv = binarizeHV(sum_hv, 0)
        AM[label] = sum_hv

        print("Finished " + str(n) + "-gram sum computing for: " + (label + ".txt"))
        file.close()

    return iM, AM


def test(model, n):
    test_files = os.listdir("language_recognition_data/testing_texts")
    language_names = {}
    language_names["af"] = "afr"
    language_names["bg"] = "bul"
    language_names["cs"] = "ces"
    language_names["da"] = "dan"
    language_names["nl"] = "nld"
    language_names["de"] = "deu"
    language_names["en"] = "eng"
    language_names["et"] = "est"
    language_names["fi"] = "fin"
    language_names["fr"] = "fra"
    language_names["el"] = "ell"
    language_names["hu"] = "hun"
    language_names["it"] = "ita"
    language_names["lv"] = "lav"
    language_names["lt"] = "lit"
    language_names["pl"] = "pol"
    language_names["pt"] = "por"
    language_names["ro"] = "ron"
    language_names["sk"] = "slk"
    language_names["sl"] = "slv"
    language_names["es"] = "spa"
    language_names["sv"] = "swe"

    stats = {} # key is name of language, value is list [correct, total number]

    for test_file in test_files:
        key = test_file.split("_")[0]
        label = language_names.get(key, None)

        if label == None:
            print("could not find " + key + " in language_names")
            continue

        file = open("language_recognition_data/testing_texts/" + test_file, "r")

        for line in file:
            query_hv = binarizeHV(gen_n_gram_sum(line, model.iM, model.D, n), 0)
            classfiication = model.query(query_hv)

            if label not in stats:
                stats[label] = [0, 0]

            correct = 0

            if classfiication == label:
                correct = 1

            stats[label][0] += correct
            stats[label][1] += 1

        file.close()


    file = open("accuracy-" + str(n) + "-n-gram.txt", "w")


    sum_acc = 0
    for lang in stats:
        num_correct = stats[lang][0]
        total_tests = stats[lang][1]
        accuracy = num_correct / total_tests
        sum_acc += accuracy
        file.write(lang + " acc: " + str(accuracy) + "\n")
        #print(lang + " acc: " + str(accuracy))

    avg_acc = sum_acc / 21
    #print("avg_acc: " + str(avg_acc))
    file.write("avg_acc: " + str(avg_acc) + "\n")

    file.close()




        




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


class HD_Model:
    def __init__(self, D):
        self.iM  = {}
        self.CiM = {}
        self.AM  = {}
        self.D = D

    def save(self, file_name):
        file = open(file_name, "wb")
        pickle.dump((self.iM, self.CiM, self.AM, self.D), file)
        file.close()

    def load(self, file_name):
        file = open(file_name, "rb")
        self = pickle.load((self.iM, self.CiM, self.AM, self.D))
        file.close()

    def train(self, func, arg=None):
        self.iM, self.AM = func(arg)

    def _cos_angle(self, hv1, hv2):
        return sum((hv1 * hv2)) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))

    def query(self, query_hv, print_all=False):
        label = "NONE"
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
        pass
        

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
    D = 10000
    n = 2

    hd_model = HD_Model(D)
    hd_model.train(learn_languages, n)
    save(hd_model, "model-n-2.bin")

    #hd_model = load("model-n-4.bin")


    #test_text = "gdybysmy znalezli sie wsrod czlonkow zalozycieli strefy euro  a powinnismy sie znalezc  nasza pozycja dzisiaj bylaby o wiele silniejsza "
    #query_hv = binarizeHV(gen_n_gram_sum(test_text, hd_model.iM, D, n), 0)
    #print(hd_model.query(query_hv, print_all=True))


    test(hd_model, n)


    #iM, AM = learn_languages(3)
    #print(find_label(query_hv, AM))


    




if __name__ == "__main__":
    main()