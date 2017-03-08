import os
import pandas as pd


class DataException(Exception):
	"""Thrown when there is an error laoding data"""
	pass


class Dataloader():

	def getTrain(self):
		"""Return the training data set"""
		return None

	def getDev(self):
		"""Return the dev data set"""
		return None

	def getTest(self):
		"""Return the test data set"""
		return None

	def getData(self):
		"""Load the data and return the train,dev and test objects"""
		train = self.getTrain()
		dev = self.getDev()
		test = self.getTest()
		print "Loaded train data set with %i examples"%len(train)
		print "Loaded dev data set with %i examples"%len(dev)
		print "Loaded test data set with %i examples"%len(test)
		return train,dev,test




class MicrosoftDataloader(Dataloader):
	train_name = "data/msrdata/msr_train.xlsx"
	dev_name = "data/msrdata/msr_test.xlsx"
	test_name = "data/msrdata/msr_test.xlsx"

	def _file_to_pandas(self,filename):
		if not os.path.exists(filename):
			raise DataException(filename+' does not exist')
		dataframe = pd.read_excel(filename, sep='\t')
		dataframe.rename(columns={
			'Quality': 'label',
			'#1 String': 'sentence1',
			'#2 String': 'sentence2'
		}, inplace=True)
		return dataframe

	def getTrain(self):
		"""Return the training data set"""
		return self._file_to_pandas(self.train_name)

	def getDev(self):
		"""Return the dev data set"""
		return self._file_to_pandas(self.dev_name)

	def getTest(self):
		"""Return the test data set"""
		return self._file_to_pandas(self.test_name)



