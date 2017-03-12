import tensorflow as tf
import numpy as np
import seq2seq
from lib.callbacks import ParaphraseCallback
from seq2seq.models import Seq2Seq

# Data loading imports
from data.loaders import MicrosoftDataloader
from data.parsers import DataParser

GLOVE_SIZE = 50
HIDDEN_SIZE = 100
HIDDEN_DEPTH = 12
MAX_SENTENCE_LENGTH = 51

# Load the data and build the vocab
loader = MicrosoftDataloader()
train, dev, test = loader.getData()
parser = DataParser(train, dev, test)
pretrained_embeddings = parser.get_embeddings_matrix()

# Parse the training data
print "Loading embedded training set"
train_labels, train_inputs = parser.get_embeddings(train,MAX_SENTENCE_LENGTH)
test_labels, test_inputs = parser.get_embeddings(test,MAX_SENTENCE_LENGTH)
dev_labels, dev_inputs = parser.get_embeddings(dev,MAX_SENTENCE_LENGTH)

# Use each item of the paraphrase pairs when training the autoencoder
# The target size is (n_training, max_sentence_length, embedding_size)
print "Reshaping training data"
X_train = np.reshape(train_inputs,(-1,MAX_SENTENCE_LENGTH,GLOVE_SIZE))
X_test = np.reshape(test_inputs,(-1,MAX_SENTENCE_LENGTH,GLOVE_SIZE))

# Build the Siamese network callback
siamese = ParaphraseCallback(train_inputs, train_labels, test_inputs, test_labels)

# Train the autoencoder
# The autoencoder has two input-output columns so it can be used to
# encode pairs of sentences in the Siamese model
print "Building RNN Autoencoder <Seq2Seq> model"
autoencoder = Seq2Seq(
	input_dim=GLOVE_SIZE,
	hidden_dim=HIDDEN_SIZE,
	output_length=MAX_SENTENCE_LENGTH,
	output_dim=GLOVE_SIZE,
	depth=HIDDEN_DEPTH
)

print "Compiling RNN Autoencoder <Seq2Seq> model"
autoencoder.compile(loss='mse', optimizer='rmsprop',loss_weights=[1., 1.])

for i in range(10):
	print "------------- Attempt %i --------------"%i
	print "Fitting RNN Autoencoder <Seq2Seq> model"
	autoencoder.fit([X_train,X_train], [X_train,X_train], nb_epoch=4, batch_size=128)#, callbacks=[siamese])

	print "Evaluating RNN Autoencoder <Seq2Seq> on training set"
	autoencoder.evaluate([X_train,X_train], [X_train,X_train], batch_size=128)

	print "Evaluating RNN Autoencoder <Seq2Seq> on test set"
	autoencoder.evaluate([X_test,X_test], [X_test,X_test], batch_size=128)

