"""
Train a logistic regression over the sentence embeddings
All of the heavy-lifting (deep learning) has been done when training the models
and encoding the sentences. Therefore we just use regular Numpy to train the logistic regression

As much as it would be great to have an end to end model, there is no really much point
because we only have ~4000 paraphrase training examples
"""
import json
import numpy as np
from utils import is_number
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from glove.data.loaders import get_paraphrase_sentences
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

GLOVE_EMBEDDINGS = "./glove/pickle/embed.json"
DOC2VEC_EMBEDDINGS = "./doc2vec/pickle/embed.json"
TRANSLATOR_EMBEDDINGS = "./translator/pickle/embed.json"
SKIPTHOUGHT_EMBEDDINGS = "./skip-thoughts-master/pickle/embed.json"


# Load glove embeddings
with open(GLOVE_EMBEDDINGS) as file:
	glove_embeddings = json.load(file)

# Load doc2vec embeddings
with open(DOC2VEC_EMBEDDINGS) as file:
	dec2vec_embeddings = json.load(file)

# Load the translator embeddings
with open(TRANSLATOR_EMBEDDINGS) as file:
	translator_embeddings = json.load(file)

# Load the skipthought embeddings
with open(SKIPTHOUGHT_EMBEDDINGS) as file:
	skipthought_embeddings = json.load(file)



def get_glove_similarity(sentence1,sentence2):
	"""Get the similarity of two sentences by their index"""
	embed1 = np.atleast_2d(glove_embeddings[sentence1]['embedding'])
	embed2 = np.atleast_2d(glove_embeddings[sentence2]['embedding'])
	cosine = np.mean(embed1*embed2)
	euclid = np.mean(embed1-embed2)
	return [cosine,euclid]


def get_doc2vec_similarity(sentence1,sentence2):
	"""Get the similarity of two sentences by their index"""
	embed1 = np.atleast_2d(dec2vec_embeddings[sentence1]['embedding'])
	embed2 = np.atleast_2d(dec2vec_embeddings[sentence2]['embedding'])
	cosine = np.mean(embed1*embed2)
	euclid = np.mean(embed1-embed2)
	return [cosine,euclid]


def get_skipthought_similarity(sentence1,sentence2):
	"""Get the similarity of two sentences by their index"""
	embed1 = np.atleast_2d(skipthought_embeddings[sentence1]['embedding'])
	embed2 = np.atleast_2d(skipthought_embeddings[sentence2]['embedding'])
	cosine = np.mean(embed1*embed2)
	euclid = np.mean(np.abs(embed1-embed2))
	return [cosine,euclid]


def get_translator_similarity(sentence1,sentence2):
	"""Get the similarity of two sentences by their index"""
	embed1 = np.atleast_2d(translator_embeddings[sentence1]['embedding'])
	embed2 = np.atleast_2d(translator_embeddings[sentence2]['embedding'])
	#cosines = []
	#euclids = []
	#for layer in [0,1024,2048]:
	#		cosines.append(cosine_distances(context1,context2))
	#	euclids.append(euclidean_distances(context1,context2))
	cosines = [np.mean(embed1*embed2)]
	euclids = [np.mean(embed1-embed2)]
	return cosines+euclids


def get_feats(tA, tB):
    """
    Compute additional features (similar to Socher et al.)
    These alone should give the same result from their paper (~73.2 Acc)
    Implimentation inspired by code from original SkipThoughts paper
    @input {List} tA. Tokens for sentence1
    @input {List} tB. Tokens for sentence2
    """
    nA = [w for w in tA if is_number(w)]
    nB = [w for w in tB if is_number(w)]

    features = np.zeros((6,))
	# n1
    if set(nA) == set(nB):
        features[0] = 1.
	# n2
    if set(nA) == set(nB) and len(nA) > 0:
        features[1] = 1.
	# n3
    if set(nA) <= set(nB) or set(nB) <= set(nA):
        features[2] = 1.
	# n4
    features[3] = 1.0 * len(set(tA) & set(tB)) / len(set(tA))
	# n5
    features[4] = 1.0 * len(set(tA) & set(tB)) / len(set(tB))
	# n6
    features[5] = 0.5 * ((1.0*len(tA) / len(tB)) + (1.0*len(tB) / len(tA)))
    return features



def main():
	# Load all of the training/testing data
	# Here we only need the sentence indices and the labels
	tokenized,pairs = get_paraphrase_sentences(fileprefix="")

	# We create two column numpy arrays
	# @labels is a column of labels
	# @inputs is a column of similarity values for each training example
	labels = np.zeros((len(pairs),1))
	inputs = np.zeros((len(pairs),14))

	for i,pair in enumerate(pairs):
		sentence1,sentence2,label = pair
		labels[i] = label
		inputs[i,0:2] = get_glove_similarity(sentence1,sentence2)
		inputs[i,2:4] = get_doc2vec_similarity(sentence1,sentence2)
		inputs[i,4:6] = get_translator_similarity(sentence1,sentence2)
		inputs[i,6:12] = get_feats(tokenized[sentence1][0], tokenized[sentence2][0])
		inputs[i,12:14] = get_skipthought_similarity(sentence1,sentence2)

	# Our last chance to drop any of the inputs
	# inputs = inputs[:,4:6]

	# Split into test and training accorind to original files
	n_training = 4076
	X_train, X_test = inputs[:n_training], labels[:n_training]
	y_train, y_test = inputs[n_training:], labels[n_training:]

	model = LogisticRegression(C=4)
	model.fit(X_train, y_train)

	print "\nStats for Training Set: "
	pred_train = model.predict(X_train)
	evaluate(y_train, pred_train)

	print "\nStats for Testing Set: "
	pred_test = model.predict(X_test)
	evaluate(y_test, pred_test)




def evaluate(labels,pred):
	"""Print a whole lot of metrics in the pred/labels combination"""
	p,r,f,_ = precision_recall_fscore_support(labels, pred, average='binary')
	print "    positive fraction: ", np.mean(pred)
	print "    negative fraction: ", np.mean(1-pred)
	print "    accuracy: ", accuracy_score(labels,pred)
	print "    precision: %.3f,  recall:  %.3f,  F1: %.3f"%(p,r,f)
	print "\n", confusion_matrix(labels, pred), "\n"
	print "--------------------------------------\n"






if __name__=="__main__":
	main()


