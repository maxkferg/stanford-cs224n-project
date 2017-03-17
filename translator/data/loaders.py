import os
import nltk
import codecs
import pandas as pd
from nltk.tokenize import word_tokenize

class DataException(Exception):
	"""Thrown when there is an error laoding data"""
	pass


class Dataloader():
	"""
	Generic loader for loading data from a file
	"""
	def __init__(self):
		"""We allow the user speficy a file prefix"""
		data = os.path.dirname(__file__)
		self.train_name = os.path.join(data,self.train_name)
		self.test_name = os.path.join(data,self.test_name)
		self.dev_name = os.path.join(data,self.dev_name)

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
	"""
	Load training sentences for MSR Corpus
	"""
	train_name = "msrdata/msr_train.xlsx"
	dev_name = "msrdata/msr_test.xlsx"
	test_name = "msrdata/msr_test.xlsx"

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
