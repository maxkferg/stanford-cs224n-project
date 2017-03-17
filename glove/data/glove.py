
import numpy as np

EMPTY_WORD = "EMPTY"
DEFAULT_FILE_PATH = "data/glove/glove.6B/glove.6B.50d.txt"


def load_word_vectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """
    Read pretrained GloVe vectors
    Return a numpy array where the row index corrosponds to token[i]
    ie word_vector[i,:] is the word vector for token[i]
    """
    word_vectors = np.zeros((len(tokens), dimensions))
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            token_index = tokens[token]
            word_vectors[token_index,:] = np.asarray(data)
    return word_vectors