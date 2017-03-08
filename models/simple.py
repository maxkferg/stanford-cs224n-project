import os
import time
import cPickle
import numpy as np
import tensorflow as tf
from lib.model import Model
from data.parsers import DataParser
from data.loaders import MicrosoftDataloader
from encoders.simple import BagOfVectorsEncoder
from utils.general_utils import Progbar,get_minibatches
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from data.parsers import MAX_LENGTH



class ClassifierConfig(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    gamma = 0
    dropout = 0.8
    n_classes = 1
    max_length = MAX_LENGTH
    embed_size = 50
    hidden_size = 20
    batch_size = 128
    n_epochs = 100
    lr = 0.003



class EncoderConfig(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_classes = 1
    max_length = MAX_LENGTH
    embed_size = 50
    hidden_size = 300
    batch_size = 32
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001



def binary_activation(x):
    """Return 0 if x is less than 1, otherwise return 1"""
    return tf.cast(x > 0, dtype=tf.int32)


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """
    d = tf.reduce_sum(tf.square(tf.sub(x, y)),1)
    return d


def compute_contrastive_loss(distance, label, margin=0.2):

    """
    Compute the contrastive loss as in


    L = 0.5 * Y * D^2 + 0.5 * (Y-1) * {max(0, margin - D)}^2

    **Parameters**
     left_feature: First element of the pair
     right_feature: Second element of the pair
     label: Label of the pair (0 or 1)
     margin: Contrastive margin

    **Returns**
     Return the loss operation

    """
    one = tf.constant(1.0)
    d_sqrt = tf.sqrt(distance)
    first_part = (one-label)*distance # (Y-1)*(d)
    max_part = tf.square(tf.maximum(margin-d_sqrt, 0))
    second_part = label*max_part # (Y) * max(margin - d, 0)
    loss = 0.5 * tf.reduce_mean(first_part + second_part)
    return loss


def contrastive_loss(distance,label):
    margin = 0.2
    d_sqrt = tf.sqrt(distance)
    loss = label * tf.square(tf.maximum(0.0, margin - d_sqrt)) + (1 - label) * distance
    loss = 0.5 * tf.reduce_mean(loss)
    return loss



class SimpleClassifier(Model):
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
        input_shape = (None, self.config.max_length,2)
        labels_shape = (None, self.config.n_classes)
        dropout_shape = ()
        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=input_shape, name="input_placeholder")
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
        xavier = tf.contrib.layers.xavier_initializer()
        W_shape = (self.config.embed_size, self.config.hidden_size)
        b_shape = (self.config.hidden_size, )

        with tf.variable_scope("embedded_vectors"):
            x1 = self.encoder.add_encoding_op(self.input_placeholder[:,:,0])
            tf.get_variable_scope().reuse_variables()
            x2 = self.encoder.add_encoding_op(self.input_placeholder[:,:,1])

        x1 = tf.layers.dropout(x1,self.dropout_placeholder)
        x2 = tf.layers.dropout(x2,self.dropout_placeholder)

        with tf.variable_scope("classifier_hidden_layer"):
            W = tf.get_variable("W", W_shape, initializer=xavier, dtype=tf.float32)
            b = tf.get_variable("b", b_shape, initializer=xavier, dtype=tf.float32)
            h1 = tf.sigmoid(tf.matmul(x1,W)+b)
            h2 = tf.sigmoid(tf.matmul(x2,W)+b)
        #return tf.reduce_mean(tf.square(h1 - h2), axis=1, keep_dims=True)
        # Compute the cosine similarity between s1 and s2
        # Axis 1 is now the enbedding-feature dimension
        a1 = tf.nn.l2_normalize(h1, dim=1)
        a2 = tf.nn.l2_normalize(h2, dim=1)
        pred = tf.reduce_sum(a1*a2, axis=1, keep_dims=True)
        #assert pred.get_shape().as_list() == [None,1], "predictions are the wrong shape. Expected [None], got {}".format(pred.get_shape().as_list())
        return pred


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
        assert pred.get_shape().as_list() == self.labels_placeholder.get_shape().as_list() , "predictions and labels should have the same size"

        ytrue = tf.cast(self.labels_placeholder,dtype=tf.float32)
        #contrastive_loss
        margin = 0.2
        error = tf.abs(pred-ytrue)
        losses = tf.nn.l2_loss(tf.maximum(error-0.2,0))
        loss = tf.reduce_mean(losses)
        return loss
        #return compute_contrastive_loss(1-pred, ytrue)



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
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op


    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def evaluate_on_batch(self, sess, inputs_batch, labels_batch):
        """Return the precision,recall,fscore for the current batch"""
        feed = self.create_feed_dict(inputs_batch, dropout=self.config.dropout)
        yhat = binary_activation(self.pred-0.5)
        pred = sess.run(yhat, feed_dict=feed)
        #print ""
        #print confusion_matrix(labels_batch, pred)
        #print ""
        p,r,f,_ = precision_recall_fscore_support(labels_batch, pred, average='binary')
        return p,r,f


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
        iterator = get_minibatches([dev_inputs, dev_labels], self.config.batch_size)
        for i, (train_x, train_y) in enumerate(iterator):
            prf.append(self.evaluate_on_batch(sess, train_x, train_y))
            prog.update(i + 1)
        prf = np.mean(np.array(prf),axis=0)
        print "Precision={:.2f}, Recall={:.2f}, F1={:.2f}".format(prf[0],prf[1],prf[2])
        return prf[2]



    def fit(self, sess, saver, train_inputs, train_labels, dev_inputs, dev_labels):
        best_f1 = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            self.run_train_epoch(sess, train_inputs, train_labels)
            f1 = self.run_dev_epoch(sess, dev_inputs, dev_labels)
            if f1 > best_f1:
                best_f1 = f1
                if saver:
                    print "New best dev F1! Saving model in ./data/weights/parser.weights"
                    saver.save(sess, './data/weights/parser.weights')
            print


    def tensorboard_summaries(self):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            tf.summary.histogram('pred', self.pred)
            tf.summary.scalar('loss', self.loss)
            merged = tf.summary.merge_all()



    def __init__(self, config, encoder):
        """Initialize the Simple Classifier with the pretrained embeddings"""
        self.encoder = encoder
        self.config = config
        self.build()
        self.tensorboard_summaries()


def main(debug=True):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    start = time.time()

    # Load and tokenize the data
    loader = MicrosoftDataloader()
    train, dev, test = loader.getData()

    # Build the sentence parser
    parser = DataParser(train, dev, test)
    pretrained_embeddings = parser.get_embeddings_matrix()

    # Parse the training data
    train_labels, train_inputs = parser.parse(train)
    test_labels, test_inputs = parser.parse(test)
    dev_labels, dev_inputs = parser.parse(dev)
    print train_inputs.shape

    with tf.Graph().as_default():
        # Build the encoder and classifier
        encoder = BagOfVectorsEncoder(EncoderConfig(), pretrained_embeddings)
        classifier = SimpleClassifier(ClassifierConfig(), encoder)

        if not os.path.exists('./data/weights/'):
            os.makedirs('./data/weights/')

        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()

        print "took %.2f seconds"%(time.time() - start)

        with tf.Session() as session:
            #writer = tf.summary.FileWriter('logs',session.graph)
            session.run(init)
            print 80 * "="
            print "TRAINING"
            print 80 * "="
            classifier.fit(session, saver, train_inputs, train_labels, dev_inputs, dev_labels)

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



















