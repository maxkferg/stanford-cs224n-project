import random
import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import *
from keras.layers import Input,merge
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix


def euclidean_distance(inputs):
    if (len(inputs) != 2):
        raise 'oops'
    output = K.mean(K.square(inputs[0] - inputs[1]), axis=-1)
    output = K.sqrt(output)
    output = K.expand_dims(output, 1)
    return output


def contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))


def compute_accuracy(predictions, labels, threshold=0.5):
    """ Compute classification accuracy with a fixed threshold on distances.
    """
    accuracy = (yhat==labels)
    return accuracy.mean()



class SiameseParaphrase():
    """Simple feedforward paraphrase classification model"""

    def __init__(self, autoencoder, input_dimension):

        # Get the inputs to the autoencoder
        inputLeft = autoencoder.inputs[0]
        inputRight = autoencoder.inputs[1]

        # Get the outputs from the autoencoder
        outputLeft = autoencoder.midpoints[0]
        outputRight = autoencoder.midpoints[1]

        # merge outputs of the base network and compute euclidean distance
        lambda_merge = merge([outputLeft, outputRight], mode=euclidean_distance, output_shape=[[None,1]])

        # create main network
        model = Model([inputLeft,inputRight],lambda_merge)

        # compile
        rms = RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms)

        # save
        self.autoencoder = autoencoder
        self.model = model

    def fit(self,train_left,train_right,labels):
        """Fit the model to the data"""
        print "Fitting Paraphrase <SiameseParaphrase> model: "
        self.model.fit([train_left, train_right], labels, batch_size=128, nb_epoch=2)

    def predict(self,x_left,x_right,threshold=0.5):
        """Predict the output labels (binary)"""
        scores = self.model.predict([x_left, x_right])
        yhat = scores < threshold # With L2 a low score -> matching sentence
        return yhat

    def evaluate(self, x_left, x_right, labels):
        # compute final accuracy on training and test sets
        yhat3 = self.predict([x_left, x_right])
        yhat5 = self.predict([x_left, x_right])
        yhat7 = self.predict([x_left, x_right])
        p,r,f,_ = precision_recall_fscore_support(labels, yhat5, average='binary')
        print('* Accuracy (0.4): %0.2f%%' % (100 * compute_accuracy(yhat3, labels, 0.4)))
        print('* Accuracy (0.5): %0.2f%%' % (100 * compute_accuracy(yhat5, labels, 0.5)))
        print('* Accuracy (0.6): %0.2f%%' % (100 * compute_accuracy(yhat7, labels, 0.6)))
        # Show the confusion matrix and precision/recall
        p,r,f,_ = precision_recall_fscore_support(labels, yhat5, average='binary')
        print "Precision={:.2f}, Recall={:.2f}, F1={:.2f}".format(p,r,f)
        print confusion_matrix(labels, yhat5)



