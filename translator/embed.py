import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from lib import data_utils
from utils import get_bucket,chunker
from data.loaders import MicrosoftDataloader
from translate import create_model, _buckets
from scipy.spatial.distance import cosine

EMBED_FILE = "./pickle/embed.json"
TRAIN_DIR = "./data/checkpoints"
DATA_DIR = "./data/vocab"
VOCAB_SIZE = 40000
BATCH_SIZE = 64



def get_sentence_to_context_map(sentences):
  """
  Process all of the sentences with the model
  Return a map between sentence text and the context vectors
  The order of the map is undefined due to the bucketing process
  """
  # Load the vocab
  en_vocab = get_english_vocab(DATA_DIR,VOCAB_SIZE)

  # Allocate the sentences to buckets
  bucketed = {}
  for sentence in sentences:
    bucket_id = get_bucket(en_vocab,sentence)
    bucketed.setdefault(bucket_id,[])
    bucketed[bucket_id].append(sentence)

  mapped = {}
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True, train_dir=TRAIN_DIR)
    model.batch_size = BATCH_SIZE  # We decode 64 sentence at a time.
    # Iterate over each bucket
    for bucket_id,sentences in bucketed.iteritems():
      for batch in chunker(sentences,BATCH_SIZE):
        data = []
        # Tokenize each sentence
        for sentence in batch:
          token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
          expected_output = []
          data.append((token_ids, expected_output))
        # Use the model to obtain contexts for each sentence in the batch
        encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: data}, bucket_id)
        contexts = model.step_context(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id)
        features = np.hstack(contexts)
        print 'Encoded {0} sentences into {1} dimensional vectors'.format(*features.shape)
        # Now we align sentences with their contexts
        for i,sentence in enumerate(batch):
          mapped[sentence] = features[i,:].tolist()
  return mapped




def main():
  """
  Encode all of the sentences to vector form
  The sentence vectors are saved to JSON along with some metadata
  The context vector is 3*1024 and the word vectors is sized accordingly
  """
  loader = MicrosoftDataloader()
  train,dev,test = loader.getData()
  sentences = []

  # Collect all the training sentences
  for i,row in pd.concat((train,test)).iterrows():
    if isinstance(row["sentence1"], basestring) and isinstance(row["sentence2"], basestring):
      sentences.append(row["sentence1"])
      sentences.append(row["sentence2"])

  # Get the mapping between sentences and their cotext vectors
  mapped = get_sentence_to_context_map(sentences)

  # At this stage we have a map between every sentence and its context vector
  # However the JSON file must contain sentences in the same order as in the MSR data file
  data = []
  for i,sentence in enumerate(sentences):
    embedding = mapped[sentence]
    data.append({'index':i, 'embedding':embedding, 'text':sentence})

  # Write the sentences and embeddings to JSON
  # The array index should corrospond to the sentence #
  print "Saving embedded sentences to: {0}".format(EMBED_FILE)
  with open(EMBED_FILE,'w') as outfile:
    json.dump(data,outfile,indent=2)




if __name__=="__main__":
  main()
