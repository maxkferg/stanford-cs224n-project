import tensorflow as tf
from data.parsers import MAX_SENTENCE_LENGTH

def get_row_counts(A):
    """Return the number nonzero elements in each row"""
    zero = tf.constant(0, dtype=tf.int32)
    where = tf.not_equal(A, zero)
    mask = tf.cast(where, tf.float32)
    return tf.reduce_sum(mask,axis=1)


class BagOfVectorsEncoder(object):
    """
    Implements a simple bag of vectors model
    Words in the sentence are converted to vectors using an embedding matrix
    The encoded sentence is the mean of the word vectors
    """

    def add_embedding_op(self,tokens):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, embed_size).

        Returns:
            embeddings: tf.Tensor of shape (None, embed_size)
        """
        with tf.variable_scope("embeddings"):
            embedding_tensor = tf.Variable(self.pretrained_embeddings,dtype=tf.float32)
            features = tf.nn.embedding_lookup(embedding_tensor, tokens)
            newshape = (-1, MAX_SENTENCE_LENGTH, self.config.embed_size)
            embeddings = tf.reshape(features, newshape)
        return embeddings



    def add_encoding_op(self,tokens):
        """Adds the unrolled RNN:
        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        word_vectors = self.add_embedding_op(tokens)
        preds = tf.reduce_sum(word_vectors,axis=1) #/ get_row_counts(tokens)

        # The preds should have shape (?,self.embed_size)
        # Each sentence in the batch is represented by a single vector with embed size
        assert preds.get_shape().as_list() == [None, self.config.embed_size], "sentence encodings are the wrong shape. Expected {}, got {}".format([None, self.config.embed_size], preds.get_shape().as_list())
        return preds


    def __init__(self, config, pretrained_embeddings):
        self.config = config
        self.pretrained_embeddings = pretrained_embeddings









