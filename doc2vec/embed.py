"""
Generate sentence vectors using the trained doc2vec model
"""
import json
import numpy as np
from data.loaders import get_paraphrase_sentences
from gensim.models import doc2vec

MODEL_FILE = './pickle/model-large.doc2vec'
EMBED_FILE = './pickle/embed.json'

print "Loading model from {0}".format(MODEL_FILE)
model = doc2vec.Doc2Vec.load(MODEL_FILE)

print "Loading paraphrase test sentences"
sentences,_ = get_paraphrase_sentences()


data = []
for i,sentence in enumerate(sentences):
	print i,sentence[1]
	assert i==sentence[1]
	docvec = model.docvecs[i].tolist()
	text = " ".join(sentence[0]) # Include for sanity only
	data.append({'index':i, 'embedding':docvec, 'text':text})

# Write the sentences and embeddings to JSON
# The array index should corrospond to the sentence #
print "Saving embedded sentences to: {0}".format(EMBED_FILE)
with open(EMBED_FILE,'w') as outfile:
	json.dump(data,outfile,indent=2)




