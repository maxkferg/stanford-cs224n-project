# Glove Bag-of-Vectors Embeddings

In this folder we embed the sentences using GloVe and bag-of-vectors
The implimentation is trivial, but it is implimented here to keep the
file structure consistent for this project

## Getting Started
Download the GloVe pretrained word vectors:

```sh
source ./data/download.sh
```

Run the embedding script to embed the MSR paraphrases
```sh
python embed.py
```

## Background
We use 50d pretrained GloVe vectors from https://nlp.stanford.edu/projects/glove/.
In some of our early experiments we attempted to retrain the word vectors along with
the paraphrase model. However, with the largest paraphrase dataset containing <10,000 pairs,
there is no meaningful way to train the model

## License
MIT
