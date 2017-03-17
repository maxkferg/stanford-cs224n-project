# Doc2Vec

In this folder we train a Doc2Vec model using the `gensim` library.
the model is trained using sentences from the DeepDive Wikipedia crawl.
This code was written to generate sentence vectors for the CS224N project.

## Getting Started

Download the Wikipedia dataset from DeepDive and extract it in the
crawls folder. `http://deepdive.stanford.edu/opendata/#wiki-wikipedia-english-edition`. Note, a single example file has been left in the is directory
for testing purposes. The `msrdata` folder should already contain xlsx files that will be used
to test the model as it is trained.

Once the data is in place run the following command to train the sentence vectors
```python
python train.py
```

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