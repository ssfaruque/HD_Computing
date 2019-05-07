import sys
import os

os.chdir('..')
hdc_path = os.getcwd()
os.chdir('fake_news')
sys.path.insert(0, hdc_path)
from open_hdc import hdc






class Fake_News_HD_Model(hdc.HD_Model):
	def __init__(self, D=10000):
		super().__init__(D)

	def train(self):
		pass

	def test(self):
		pass

	def load_dataset(self):
		with open('datasets/liar_dataset/train.tsv', 'r') as file:
			lines = file.readlines()
			for line in lines:
				line = line.rstrip()
				columns = line.split('\t')
				self.dataset.append(columns)






def main():
	print("hello")

	Fake_News_Model = Fake_News_HD_Model(D=10000)

	Fake_News_Model.load_dataset()

	print(Fake_News_Model.dataset[1])





if __name__ == '__main__':
	main()