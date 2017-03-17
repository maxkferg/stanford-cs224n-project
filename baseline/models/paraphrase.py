import random
import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import *
from keras.layers import Input,merge
from keras.optimizers import SGD, RMSprop
from keras import backend as K


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


def create_pairs(x, digit_indices):
    """ Positive and negative pair creation.
        Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def compute_accuracy(predictions, labels):
    """ Compute classification accuracy with a fixed threshold on distances.
    """
    labels = predictions < 0.5
    accuracy = (labels==predictions)
    return accuracy.mean()



class SiameseParaphrase():
    """Simple feedforward paraphrase classification model"""

    def __init__(self, autoencoder,input_dimension):

        # Get the inputs to the autoencoder
        inputLeft = autoencoder.inputs[0]
        inputRight = autoencoder.inputs[1]

        # Get the outputs from the autoencoder
        print autoencoder.midpoints
        outputLeft = autoencoder.midpoints[0]
        outputRight = autoencoder.midpoints[1]

        # merge outputs of the base network and compute euclidean distance
        lambda_merge = merge([outputLeft, outputRight], mode=euclidean_distance, output_shape=[[None,1]])

        # create main network
        #model = Sequential()
        #model.add(lambda_merge)
        model = Model([inputLeft,inputRight],lambda_merge)

        # compile
        rms = RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms, loss_weights=[1., 1.])

        # display
        print model.summary()

        # save
        self.autoencoder = autoencoder
        self.model = model

    def fit(self,train_left,train_right,labels):
        """Fit the model to the data"""
        print "Fitting SiameseParaphrase model: "
        self.model.fit([train_left, train_right], labels, batch_size=128, nb_epoch=10)

    def evaluate(self, x_left, x_right, labels):
        # compute final accuracy on training and test sets
        pred = self.model.predict([x_left, x_right])
        accuracy = compute_accuracy(pred, labels)
        print('* Accuracy: %0.2f%%' % (100 * accuracy))



