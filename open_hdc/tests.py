import hdc
import numpy as np
import pprint
import collections

def test_gen_rand_hv():
	hv1 = hdc.gen_rand_hv(2)
	hv2 = hdc.gen_rand_hv(6)
	hv3 = hdc.gen_rand_hv(10)
	hv4 = hdc.gen_rand_hv(10000)

	hv_list = [hv1, hv2, hv3, hv4]

	for hv in hv_list:
		counts = collections.Counter(hv)
		if (counts[1] != len(hv) / 2) or (counts[-1] != len(hv) / 2):
			print("gen_rand_hv *** FAILED ***")

	print("gen_rand_hv *** PASSED ***")




def test_gen_item_mem():
	names = ["e1", "e2", "e3", "e4"]
	item_mem = hdc.gen_item_mem(names, 20)

	vecs = []

	for key, hd_vec in item_mem.items():

		for vec in vecs:
			if np.array_equal(vec, hd_vec):
				print("gen_item_mem *** FAILED ***")

		vecs.append(hd_vec)

	print("gen_item_mem *** PASSED ***")



def test_gen_cont_item_mem():
	cont_item_mem = hdc.gen_cont_item_mem(0, 10, 10, 6)

	pp = pprint.PrettyPrinter(width=41, compact=True)


	pp.pprint(cont_item_mem)





def main():
	#test_gen_rand_hv()
	#test_gen_item_mem()
	test_gen_cont_item_mem()


if __name__ == "__main__":
	main()
