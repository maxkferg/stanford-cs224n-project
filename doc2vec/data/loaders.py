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





class SentenceLoader(Dataloader):
	"""
	Load training sentences for paraphrase2txt
	"""
	train_name = "data/crawls/task_{0}.warc"

	def getData(self,n):
		"""Return a list of sentences from the input file"""
		words = []
		sentences = []
		count = 0
		for i in range(1,n):
			filename = self.train_name.format(i)
			print "Loading sentences from {0}".format(filename)
			with codecs.open(filename, "r", "utf-8") as file:
				for line in file:
					line = line.split()
					# If the line has words then we keep the relevant word
					if len(line)>2:
						words.append(line[1])
						continue
					# If the sentence has no words we ignore it
					if not len(words):
						continue
					# We are finished with the current word
					sentences.append([words, -1])
					# Reset the state
					words = []
					count += 1
		# Check that the sentences look good
		return sentences




def get_crawl_sentences(n):
    """
    Return tokenized sentences from the web crawler
	The return value is a list of (tokens,count) tuples
	@param n {Int}. The number of Wikipedia crawls to include
    """
    sentences = SentenceLoader().getData(n)
    print "Loaded %i sentences from the crawler"%len(sentences)
    return sentences



def get_paraphrase_sentences():
    """
    Return tokenized sentences from the MSR paraphrase database
	The return value is a list of (tokens,count) tuples
    """
    count = 0
    pairs = []
    paraphrases = []
    train,dev,test = MicrosoftDataloader().getData()

    for i,row in pd.concat((train,test)).iterrows():
        try:
            sentence1 = word_tokenize(row["sentence1"])
            sentence2 = word_tokenize(row["sentence2"])
        except AttributeError as e:
            #print e
            continue
        except TypeError as e:
            #print e
            continue
        # Now we can add both sentences to the set
        pairs.append((count, count+1, row["label"]))
        # Push them!
        paraphrases.append((sentence1, count)); count+=1
        paraphrases.append((sentence2, count)); count+=1

    print "Loaded %i sentences from Microsoft paraphrase corpus"%len(paraphrases)
    return paraphrases,pairs