import os
import json
import logging
import data_utils
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import chunker, get_bucket, get_context, get_english_vocab
from paraphrase.data.loaders import MicrosoftDataloader
from translate import create_model, _buckets
from scipy.spatial.distance import cosine

JSON_NAME = "project-data/checkpoints/sentences.json"
TRAIN_DIR = "project-data/checkpoints"
DATA_DIR = "project-data/vocab"
VOCAB_SIZE = 40000
BATCH_SIZE = 64

loader = MicrosoftDataloader()
loader.train_name = "paraphrase/data/msrdata/msr_train.xlsx"
loader.dev_name = "paraphrase/data/msrdata/msr_test.xlsx"
loader.test_name = "paraphrase/data/msrdata/msr_test.xlsx"


def encode():
  """Encode all of the sentences to vector form"""
  train,dev,test = loader.getData()
  sentences = []
  tokens = []

  # Load the vocab
  en_vocab = get_english_vocab(DATA_DIR,VOCAB_SIZE)

  # Collect all the training sentences
  for i,row in pd.concat((train,test)).iterrows():
    if isinstance(row["sentence1"], basestring) and isinstance(row["sentence2"], basestring):
      sentences.append(row["sentence1"])
      sentences.append(row["sentence2"])

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
        for sentence in batch:
          token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
          expected_output = []
          data.append((token_ids, expected_output))
        encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: data}, bucket_id)
        contexts = model.step_context(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id)
        features = np.hstack(contexts)
        print 'Extracted another set of features with shape:',features.shape
        # Now we align sentences with their contexts
        for i,sentence in enumerate(batch):
          mapped[sentence] = features[i,:].tolist()
  print sentence
  print mapped[sentence]
  print "Saving sentences to %s"%JSON_NAME
  with open(JSON_NAME,'w') as file:
    json.dump(mapped,file)





def compare():
  """Compare two sentences separated by a semi-colon"""
  # Load the data frame
  train,dev,test = loader.getData()

  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True, train_dir=TRAIN_DIR)
    model.batch_size = 64  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab = get_english_vocab(DATA_DIR,VOCAB_SIZE)

    results = []
    for i,row in train.iterrows():
      try:
        context1 = get_context(sess,model,en_vocab,row["sentence1"])[0]
        context2 = get_context(sess,model,en_vocab,row["sentence2"])[0]
      except TypeError:
        print "Error on line %i"%i
        continue
      cosine_distance = cosine(context1,context2)
      euclid_distance = np.linalg.norm(context1-context2)
      prediction = euclid_distance<10
      correctness = prediction==row["label"]
      results.append(correctness)
      print "%i,  %i,   %.3f"%(row["label"],prediction,euclid_distance)
      # Print the accuracy so far
      if i%10==0:
        print "Correctness:",np.mean(results)
    results = np.array(results)
    print np.mean(results)




if __name__=="__main__":
  encode()

