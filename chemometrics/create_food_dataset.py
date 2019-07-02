import os
import sys
import csv



def main():
	""" First command line parameter is the path to the directory containing the directories of the samples
	"""
	csvWriter = csv.writer(open('datasets/aggregate_data.csv', 'w', newline=''), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

	header_flag = False

	path = sys.argv[1]

	directories = [x[0] for x in os.walk(path)]
	directories = directories[1:]


	for directory in directories:
		category = directory.split("/")[3]
		names = os.listdir(directory)

		for name in names:
			file_path = directory + "/" + name
			file_name = category + "/" + name


			with open(file_path, "r") as file:
				ppm = name.split("-")[0]
				print(file_path)
				data = file.readlines()
				num_lines = len(data)
				wavenums = []
				absorbances = []

				for i in range(4, num_lines):
					columns = data[i].strip().split("\t")
					wavenum   = columns[0].strip()
					absorbance = columns[1].strip()

					wavenums.append(wavenum)
					absorbances.append(absorbance)


				if header_flag == False:
					csvWriter.writerow(["Label", "FileName"] + wavenums)
					header_flag = True

				csvWriter.writerow([ppm] + [file_name] + absorbances)



if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python3 create_food_dataset.py path_pointing_to_directories_of_samples")
		exit()
	main()