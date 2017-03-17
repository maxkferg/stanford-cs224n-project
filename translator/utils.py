import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from lib import data_utils

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def get_bucket(en_vocab,sentence):
  """
  Return the bucket_id that the sentence belongs to
  """
  token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
  # Which bucket does it belong to?
  bucket_id = len(_buckets) - 1
  for i, bucket in enumerate(_buckets):
    if bucket[0] >= len(token_ids):
      bucket_id = i
      break
  else:
    logging.warning("Sentence truncated: %s", sentence)
  return bucket_id



def get_english_vocab(directory,vocab_size):
  """
  Return the English vocabulary file that was used to train this model
  """
  en_vocab_path = os.path.join(directory,"vocab%d.from" % vocab_size)
  en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
  return en_vocab



def get_context(sess,model,en_vocab,sentence):
  """
  Return the context vector for the sentence
  """
  token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
  # Which bucket does it belong to?
  bucket_id = get_bucket(en_vocab,sentence)

  # Get a 1-element batch to feed the sentence to the model.
  encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)

  # Get the output context vector
  return model.step_context(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id)



def chunker(seq, size):
  """
  Chunk a list into @size blocks for iteration
  """
  return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))
