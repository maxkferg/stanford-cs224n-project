import tensorflow as tf
import numpy as np
import seq2seq
from seq2seq.models import Seq2Seq

# Data loading imports
from data.loaders import MicrosoftDataloader
from data.parsers import DataParser

# Load the data and build the vocab
loader = MicrosoftDataloader()
train, dev, test = loader.getData()
parser = DataParser(train, dev, test)
pretrained_embeddings = parser.get_embeddings_matrix()

# Parse the training data
train_labels, train_inputs = parser.parse(train)
test_labels, test_inputs = parser.parse(test)
dev_labels, dev_inputs = parser.parse(dev)


print train_labels
print train_inputs




SEQ = [
	[(2,2,2), (1,1,2,2)],
	[(2,2,3), (1,1,2,2)],
	[(2,2,5), (1,1,2,2)],
	[(2,2,6), (1,1,2,2)],
	[(2,2,1), (1,1,2,2)],
	[(2,2,2), (1,1,2,2)],
	[(2,2,8), (1,1,2,2)],
	[(2,2,6), (1,1,2,2)],
	[(2,2,7), (1,1,2,2)],
	[(2,2,0), (1,1,2,2)],
	[(1,1,2), (1,1,1,1)],
	[(1,1,4), (1,1,1,1)],
	[(1,1,1), (1,1,1,1)],
	[(1,1,2), (1,1,1,1)],
	[(3,3,2), (1,0,0,0)],
	[(3,3,4), (1,0,0,0)],
	[(3,3,5), (1,0,0,0)],
	[(3,3,3), (1,0,0,0)],
	[(3,3,3), (1,0,0,0)],
	[(3,3,3), (1,0,0,0)]
]

X_train = np.array([row[0] for row in SEQ])
Y_train = np.array([row[1] for row in SEQ])

# Add the time dimension
# Seq2Seq expects (nb_samples, timesteps, input_dim)
X_train = np.reshape(X_train, (20,3,1))
Y_train = np.reshape(Y_train, (20,4,1))

model = Seq2Seq(input_dim=1, hidden_dim=4, output_length=4, output_dim=1, depth=4)
model.compile(loss='mse', optimizer='rmsprop')
#model.fit(X_train, Y_train, nb_epoch=500, batch_size=5)



#print 'MIDPOINT:'
#print model.midpoint.predict(X_train,batch_size=20)
#print 'END'


#print model.evaluate(X_train, Y_train, batch_size=20)
#print model.predict(X_train,batch_size=20)