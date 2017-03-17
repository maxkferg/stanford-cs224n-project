import os
import pandas as pd
import numpy as np
import tensorflow as tf
from glove import load_word_vectors

NULL_WORD = "NULL_WORD"
MAX_SENTENCE_LENGTH = 52


def embedding_lookup(embeddings,indices):
	"""
	Implimentation of tf.nn.embedding_lookup with np
	If indices has dimension (a,b,c) and embeddings has dimensions
	(x,y) then the result will have dimensions (a,b,c,y)
	"""
	i_shape = list(indices.shape)
	e_shape = list(embeddings.shape[1:])
	indices = indices.flatten()
	output = embeddings[indices]
	return np.reshape(output, i_shape+e_shape)


class DataParser(object):

	def __init__(self,train,test,dev):
		"""Create a parser with a containing all words in the sets"""
		print "Building vocabulary"
		vocabulary = set([NULL_WORD])

		for dataframe in [train,test,dev]:
			for i,row in dataframe.iterrows():
				sentence1 = `row["sentence1"]`.split()
				sentence2 = `row["sentence2"]`.split()
				for word in sentence1+sentence2:
					vocabulary.add(word)
		# Create the final vocabulary for this parser
		# The index of the word in self.vocabulary represents each word
		self.vocabulary = dict([(word,i) for (i,word) in enumerate(vocabulary)])
		self.embeddings_matrix = load_word_vectors(self.vocabulary)

	def get_embeddings_matrix(self):
		"""Return the embeddings matrix"""
		return self.embeddings_matrix

	def word_to_onehot(self,word):
		"""Return the one-hot representation of a word [the index]"""
		word = word.strip('"\'!?., ').lower()
		if word in self.vocabulary:
			return self.vocabulary[word]
		return self.vocabulary[NULL_WORD]

	def parse(self,dataframe,max_length=MAX_SENTENCE_LENGTH):
		"""
		Convert the dataframe into labels and inputs
		an array with dimensions (num_examples x MAX_LENGTH x 2)
		Each row in the matrix corrosponds to a single sentence
		The third dimension of the matrix corrosponds to the sentence number (1 or 2)
		The token 0 is reserved for the absense of a word
		"""
		print "Parsing dataframe"
		labels_shape = (len(dataframe),1)
		inputs_shape = (len(dataframe),max_length,2)

		labels = np.zeros(labels_shape,dtype=np.int32)
		inputs = np.zeros(inputs_shape,dtype=np.int32)
		for i,row in dataframe.iterrows():
			labels[i] = `row["label"]`
			sentence1 = `row["sentence1"]`.split()
			sentence2 = `row["sentence2"]`.split()
			for j,sentence in enumerate([sentence1,sentence2]):
				for k,word in enumerate(sentence):
					if k>=max_length:
						print "Ignoring word %i"%k
						continue
					inputs[i,k,j] = self.word_to_onehot(word)
		return labels,inputs

	def get_embeddings(self,dataframe,max_sentence_length=MAX_SENTENCE_LENGTH):
		"""
		Similar to parse, but the returns a three dimensional inputs tensor
		@input {String} type. 'constant','variable','numpy'
		@output Tensor(num_examples) labels. The binary parahrase labels
		@output Tensor(num_examples, max_sentence_length, embedding_length, 2)
		"""
		labels,indices = self.parse(dataframe,max_sentence_length)
		params = self.get_embeddings_matrix()
		print indices.shape
		embeddings = embedding_lookup(params,indices)
		print embeddings.shape
		embeddings = np.transpose(embeddings, (0, 1, 3, 2));
		return labels,embeddings






