# Tensorflow English-French Translator

In this folder we use the encoder from an Engligh-French RNN translator as a sentence encoder.


## Background
In theory, the context layer of the RNN neural network contains a significant amount of semantic
and linguistic information about the sentence. Therefore, we train a state-of-the-art English-French
translator using an RNN network with attention. The training process takes several days on a Tesla K80 GPU.
After training the network we use the encoder to generate 3024 dimensional sentence vectors.


## Training
The model is trained as a translator using the Gigaword French-English dataset.
The model can trained using the following command:
```
python translate.py --data_dir ./data/vocab --train_dir ./data/checkpoints
```
This will automatically download the dataset to the `data_dir`
The dataset will then be tokenized and the translator will be trained
Model checkpoints will be saved in the `train_dir`

## Embedding
Sentence vectors can be embedded:
```sh
python embed.py
```

## Testing
There are two ways to test the trained model.
The first is to test the transalation (English-French) capability of the model

```sh
python translate.py --decode --data_dir ./data/vocab --train_dir ./data/checkpoints
```

Alternatively, we can use the encoder to test the similarity of two sentences.

```sh
python translate.py --compare --data_dir ./data/vocab --train_dir ./data/checkpoints
```
This will ask the user to enter two English sentences, and will return the cosine
similarity of these two sentences.


# Credits
The RNN code (in the `lib` folder) has been adapted from the Tensorflow tutorial on
machine translation. Therefore, we ask that any credit for code in the `lib` folder
be passed onto the Tensorflow authors.


## Requirements
Requirements are listed in the requirements.txt folder
You will need:

* numpy
* sklearn
* gensim
* nltk
* pandas
* scipy
* tensorflow

