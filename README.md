# CS 224N Project


## Background

In this project we evaluate a number of methods for predicting paraphrases on the
Microsoft Research Corpus. As the corpus only contains ~4000 training examples we are
unable to train a deep learning model on the corpus directly. Therefore, our approach is to train
a variety of different sentence vector models on large unrelated datasets such as the Wikipedia and Common Crawl datasets. In particular, we train the following sentence vector models:

* GloVe bag of vectors model (https://nlp.stanford.edu/projects/glove/)
* Skip Thought Vectors (https://arxiv.org/abs/1506.06726)
* Doc2vec sentence to vector model (https://arxiv.org/abs/1405.4053)
* Deep RNN language model from English-French translator

Each sentence model provides a way to encode a sentence as a vector. We find that each model is able to capture different linguistic and semantic sentence properties.
We then use each encoder to form a Siamese model with the cosine similarity metric.
A simple Logistic regression classifyer is trained on top of the cosine similarities and evaluated on the testing data set.

## Organisation
Each directory contains a separate encoding `model`.
Each model directory contains a `train.py` file which is used to generate the model (Can take days on a GPU). The model directory also contains a `generate.py` file which uses the model to generate
sentence vectors for the paraphrase vectors and save then to a `msr.json` file



## Requirements
Requirements are listed in the requirements.txt folder
You will need:

* numpy
* sklearn
* gensim
* nltk
* pandas
* scipy

## License
MIT