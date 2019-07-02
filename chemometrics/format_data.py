import glob
import os



def main():
	dataset_path = "/Users/Sahil_Faruque/Downloads/DNA_JUNE_2018"
	folder_name = "ECOLI"
	out_file = open("datasets/food_dataset_DNA_ECOLI.csv", "w")

	for file in os.listdir(dataset_path + "/" + folder_name):
		if file.endswith(".txt"):
			file_path = os.path.join(dataset_path + "/" + folder_name, file)
			print(file)

			ppm = file.split("-")[0].strip()

			with open(file_path, "r") as data_file:
				data = data_file.readlines()
				num_lines = len(data)

				for i in range(4, num_lines):
					columns    = data[i].strip().split("\t")
					wave_num   = columns[0].strip()
					absorbance = columns[1].strip()

					out_file.write(ppm + "," + wave_num + "," + absorbance + "\n")


	out_file.close()





if __name__ == "__main__":
	main()