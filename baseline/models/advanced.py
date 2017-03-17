import os
import json
import time
import cPickle
import numpy as np
import tensorflow as tf
from lib.model import Model
from data.loaders import MicrosoftDataloader
from encoders.simple import BagOfVectorsEncoder
from utils.general_utils import Progbar,get_minibatches
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix


# Load the embedded sentences (features) from file
filename = "../project-data/checkpoints/sentences.json"
with open(filename) as file:
    features = json.load(file)
print "Loaded sentence embeddings (features) from %s"%filename


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    gamma = 0
    dropout = 0.5
    n_classes = 1
    sentence_vector_size = 3*1024
    hidden_size = 20
    batch_size = 64
    n_epochs = 100
    lr = 0.0001



def binary_activation(x):
    """Return 0 if x is less than 1, otherwise return 1"""
    return (x > 0).astype(int)


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """
    d = tf.reduce_sum(tf.square(x-y),axis=1,keep_dims=True)
    return d


def euclidian_contrastive_loss(distance,label,margin=0.8):
    """
    If label==0 the distance should be larger than margin
    If label==1 the distance should be zero
    """
    loss = (1-label) * tf.square(tf.maximum(0.0, margin - distance)) + label * distance
    loss = 0.5 * tf.reduce_mean(loss)
    return loss


def cosine_contrastive_loss(pred,label,margin=0.1):
    """
    If label==0 the pred should be 0
    If label==1 the pred should be 1
    """
    error = tf.abs(pred-label)
    penalty = tf.maximum(0.0, error - margin)
    loss = 0.5 * tf.reduce_mean(penalty)
    return loss


class AdvancedModel(Model):
    """
    Compare two sentences using a simple feedforward neural network
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder
        """
        input_shape = (None, self.config.sentence_vector_size,2)
        labels_shape = (None, self.config.n_classes)
        dropout_shape = ()
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape, name="input_placeholder")
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=labels_shape, name="labels_placeholder")
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=dropout_shape, name="dropout_placeholder")


    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }


        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {}
        if inputs_batch is not None:
            feed_dict[self.input_placeholder] = inputs_batch
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        return feed_dict


    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        Note that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        Use the initializer from q2_initialization.py to initialize W and U (you can initialize b1
        and b2 with zeros)

        Hint: Here are the dimensions of the various variables you will need to create
                    W:  (n_features*embed_size, hidden_size)
                    b1: (hidden_size,)
                    U:  (hidden_size, n_classes)
                    b2: (n_classes)
        Hint: Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """
        initializer = tf.constant_initializer(0)
        W_shape = (1,self.config.sentence_vector_size)
        b_shape = (1,self.config.sentence_vector_size)

        # Separate the layers two senteces again
        # The input placeholder has dimensions (batch_size,n_features,n_sentences)
        x1 = self.input_placeholder[:,:,0]
        x2 = self.input_placeholder[:,:,1]

        with tf.variable_scope("classifier_hidden_layer"):
            W = tf.get_variable("W", W_shape, initializer=initializer, dtype=tf.float32)
            # b = tf.get_variable("b", b_shape, initializer=initializer, dtype=tf.float32)
            # Adjust the weights of each value
            a1 = x1*W
            a2 = x2*W

        # Compute the cosine similarity between s1 and s2
        # Axis 1 is now the enbedding-feature dimension
        #a1 = tf.nn.l2_normalize(a1, dim=1)
        #a2 = tf.nn.l2_normalize(a2, dim=1)
        #print a2.get_shape().as_list()
        #pred = tf.reduce_sum(a1*a2, axis=1, keep_dims=True)
        #print pred.get_shape().as_list()
        #d = tf.reduce_((a1 - a2)**2, axis=1)
        print a1.get_shape().as_list()
        distance = compute_euclidean_distance(a1,a2)
        print distance.get_shape().as_list()
        return distance


    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        with tf.variable_scope("classifier_hidden_layer",reuse=True):
            W = tf.get_variable("W", dtype=tf.float32)
            #b = tf.get_variable("b", dtype=tf.float32)
            regularization = 0#tf.nn.l2_loss(W)

        assert pred.get_shape().as_list() == self.labels_placeholder.get_shape().as_list() , "predictions and labels should have the same size"

        ytrue = tf.cast(self.labels_placeholder,dtype=tf.float32)

        return euclidian_contrastive_loss(pred, ytrue, margin=2.2)+regularization



    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer().minimize(loss)
        return train_op


    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def evaluate_on_batch(self, sess, inputs_batch, labels_batch):
        """Return the precision,recall,fscore for the current batch"""
        feed = self.create_feed_dict(inputs_batch, dropout=self.config.dropout)
        pred = sess.run(self.pred, feed_dict=feed)
        yhat = 1-binary_activation(pred-0.4)
        print pred
        #print "\n lbls mean: ",np.mean(yhat)
        #print ""
        #print confusion_matrix(labels_batch, pred)
        #print ""
        a = np.mean(yhat==labels_batch)
        p,r,f,_ = precision_recall_fscore_support(labels_batch, yhat, average='binary')
        return p,r,f,a


    def run_train_epoch(self, sess, train_inputs, train_labels):
        # Iterate through the train inputs, and train the weights
        prog = Progbar(target=1 + len(train_labels) / self.config.batch_size)
        iterator = get_minibatches([train_inputs, train_labels], self.config.batch_size)
        for i, (train_x, train_y) in enumerate(iterator):
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])


    def run_dev_epoch(self, sess, dev_inputs, dev_labels):
        # Iterate through the dev inputs and print the accuracy
        prf = []
        print "Evaluating on dev set"
        prog = Progbar(target=1 + len(dev_labels) / self.config.batch_size)
        batch_size = len(dev_labels)
        iterator = get_minibatches([dev_inputs, dev_labels], batch_size)
        for i, (train_x, train_y) in enumerate(iterator):
            prf.append(self.evaluate_on_batch(sess, train_x, train_y))
            prog.update(i + 1)
        prf = np.mean(np.array(prf),axis=0)
        print "Precision={:.2f}, Recall={:.2f}, F1={:.2f}, A={:.2f}".format(prf[0],prf[1],prf[2],prf[3])
        return prf[2]


    def fit(self, sess, saver, train_inputs, test_inputs):
        """
        Fit the model to the data
        @train_inputs should be a list of (input,label) pairs
        @test_inputs should be a list of (input,label) pairs
        """
        train_labels = np.array([x[1] for x in train_inputs])
        test_labels = np.array([x[1] for x in test_inputs])
        # Turn the labels into column vectors
        train_labels = train_labels[:, np.newaxis]
        test_labels = test_labels[:, np.newaxis]
        # The other methods expect inputs and labels to be separated
        # Right now each train input has dimensions (sentence_vector_size,2)
        # We want the input tensor to have dimensions (n_samples,sentence_vector_size,2)
        new_axis = 0
        train_inputs = np.stack([x[0] for x in train_inputs],new_axis)
        test_inputs = np.stack([x[0] for x in test_inputs],new_axis)
        #train_inputs = np.stack([x[0] for x in train_inputs],new_axis)
        #test_inputs = np.stack([x[0] for x in test_inputs],new_axis)
        # Check the shape before feeding into tensorflow
        print "training input dimensions: ", train_inputs.shape
        print "training labels dimensions: ", train_labels.shape
        print "testing input dimensions: ", test_inputs.shape
        print "testing labels dimensions: ", test_labels.shape
        assert train_inputs.shape[0]==len(train_labels)
        assert train_inputs.shape[1]==self.config.sentence_vector_size
        assert train_inputs.shape[2]==2

        # Train the model!
        best_f1 = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            self.run_train_epoch(sess, train_inputs, train_labels)
            f1 = self.run_dev_epoch(sess, test_inputs, test_labels)
            if f1 > best_f1:
                best_f1 = f1
                if saver:
                    print "New best dev F1! Saving model in ./data/weights/parser.weights"
                    saver.save(sess, './data/weights/parser.weights')


    def __init__(self, config):
        """Initialize the Simple Classifier"""
        self.config = config
        self.build()


def main(debug=True):
    print 80 * "="
    print "INITIALIZING ADVANCED MODEL"
    print 80 * "="
    start = time.time()

    # Load the data
    loader = MicrosoftDataloader()
    train, dev, test = loader.getData()

    # Create tuples of (sentence_vectors,label)
    train_inputs = []
    for index, row in train.iterrows():
        sentence_vector1 = features.get(row['sentence1'])
        sentence_vector2 = features.get(row['sentence2'])
        if (sentence_vector1 and sentence_vector2):
            sentence_vectors = np.vstack((sentence_vector1,sentence_vector2)).T
            train_inputs.append((sentence_vectors, row['label']))
        else:
            print "Skipped train row %i"%index

    # Create three tuples of (sentence_vectors,label)
    test_inputs = []
    for index, row in test.iterrows():
        sentence_vector1 = features.get(row['sentence1'])
        sentence_vector2 = features.get(row['sentence2'])
        if (sentence_vector1 and sentence_vector2):
            sentence_vectors = np.vstack((sentence_vector1, sentence_vector2)).T
            test_inputs.append((sentence_vectors, row['label']))
        else:
            print "Skipped test row %i"%index


    with tf.Graph().as_default():
        # Build the encoder and classifier
        model = AdvancedModel(Config())

        if not os.path.exists('./data/weights/'):
            os.makedirs('./data/weights/')

        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()

        print "took %.2f seconds"%(time.time() - start)

        with tf.Session() as session:
            session.run(init)
            print 80 * "="
            print "TRAINING"
            print 80 * "="
            model.fit(session, saver, train_inputs, test_inputs)

            if not debug:
                print 80 * "="
                print "TESTING"
                print 80 * "="
                print "Restoring the best model weights found on the dev set"
                saver.restore(session, './data/weights/parser.weights')
                print "Final evaluation on test set",
                print "- test UAS: {:.2f}".format(UAS * 100.0)
                print "Writing predictions"
                with open('q2_test.predicted.pkl', 'w') as f:
                    cPickle.dump(dependencies, f, -1)
                print "Done!"

if __name__ == '__main__':
    main(debug=False)



















