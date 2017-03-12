import keras
from models.paraphrase import SiameseParaphrase

class ParaphraseCallback(keras.callbacks.Callback):
    """
    Run the paraphrase model to test accuracy
    @input_shape = (max_sentence_length, embedding_size)
    """

    def __init__(self,train_inputs,train_labels,test_inputs,test_labels,input_shape):
        self.train_left = train_inputs[:,:,:,0]
        self.train_right = train_inputs[:,:,:,1]
        self.train_labels = train_labels

        self.test_left = test_inputs[:,:,:,0]
        self.test_right = test_inputs[:,:,:,1]
        self.test_labels = test_labels
        self.input_shape = input_shape


    def on_batch_end(self, batch, logs={}):
        autoencoder = self.model
        siamese = SiameseParaphrase(autoencoder,self.input_shape)
        print "Fitting SiameseModel:"
        siamese.fit(self.train_left, self.train_right ,self.train_labels)

        print "Evaluating SiameseModel in training data:"
        siamese.evaluate(self.train_left, self.train_right, self.train_labels)

        print "Evaluating SiameseModel in testing data:"
        siamese.evaluate(self.test_left, self.test_right, self.test_labels)
