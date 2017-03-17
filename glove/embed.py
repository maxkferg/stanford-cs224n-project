"""
Generate sentence vectors using the trained doc2vec model
"""
import json
import numpy as np
from data.glove import load_word_vectors
from data.loaders import get_paraphrase_sentences
from gensim.models import doc2vec

DIMENSIONS = 50
GLOVE_FILE = './data/glove/glove.6b/glove.6B.{0}d.txt'.format(DIMENSIONS)
EMBED_FILE = './pickle/embed.json'


# Load the paraphrases
print "Loading paraphrase test sentences"
sentences,_ = get_paraphrase_sentences()


# Build the vocab
print "Building paraprase vocab"
vocab = set()
for sentence in sentences:
	tokens = sentence[0]
	vocab = vocab.union(tokens)
vocab = dict([(k,i) for i,k in enumerate(vocab)]) # Indices are important
print "Vocab constructed with {0} words".format(len(vocab))


# Get all the word vectors
print "Loading GloVe word vectors from {0}".format(GLOVE_FILE)
word_vectors = load_word_vectors(vocab,GLOVE_FILE,dimensions=DIMENSIONS)
print word_vectors.shape

# Create the bag-of-vectors
data = []
for i,sentence in enumerate(sentences):
	assert i==sentence[1]
	token_indices = [vocab[t] for t in sentence[0]]
	bag_of_vectors = [word_vectors[i,:] for i in token_indices]
	embedding = np.mean(bag_of_vectors, axis=0).tolist()
	text = " ".join(sentence[0]) # Include for sanity only
	data.append({'index':i, 'embedding':embedding, 'text':text})

# Write the sentences and embeddings to JSON
# The array index should corrospond to the sentence #
print "Saving embedded sentences to: {0}".format(EMBED_FILE)
with open(EMBED_FILE,'w') as outfile:
	json.dump(data,outfile,indent=2)




