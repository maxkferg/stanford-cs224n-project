import keras
from models.paraphrase import SiameseParaphrase

class ParaphraseCallback(keras.callbacks.Callback):
    """Run the paraphrase model to test accuracy"""

    def __init__(self,train_inputs,train_labels,test_inputs,test_labels):
        self.train_left = train_inputs[:,:,:,0]
        self.train_right = train_inputs[:,:,:,1]
        self.train_labels = train_labels

        self.test_left = test_inputs[:,:,:,0]
        self.test_right = test_inputs[:,:,:,1]
        self.test_labels = test_labels


    def on_batch_end(self, batch, logs={}):
        autoencoder = self.model
        print self.train_left
        print self.train_right
        input_shape = (51,50) # (n_training, max_sentence_length, embedding_size)
        siamese = SiameseParaphrase(autoencoder,input_shape)
        print "Fitting SiameseModel:"
        siamese.fit(self.train_left, self.train_right ,self.train_labels)

        print "Evaluating SiameseModel in training data:"
        siamese.evaluate(self.train_left, self.train_right, self.train_labels)

        print "Evaluating SiameseModel in testing data:"
        siamese.evaluate(self.test_left, self.test_right, self.test_labels)
