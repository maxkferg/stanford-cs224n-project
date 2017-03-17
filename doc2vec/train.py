import numpy as np
from random import shuffle
from pprint import pprint
from gensim.models import doc2vec
from data.loaders import get_crawl_sentences,  get_paraphrase_sentences
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from gensim.models.doc2vec import LabeledSentence


# Choose the number of wikipedia crawls to use
# Each crawl contains about 18 Mb of text
N_CRAWLS = 150
MODEL_FILE = './pickle/model-epoch-{0}.doc2vec'
MODEL_SAVE_DURATION = 5 # epochs


# Use our loader to get some sentences
paraphrases, pairs = get_paraphrase_sentences()
crawl = get_crawl_sentences(n=N_CRAWLS)
sentences = paraphrases+crawl


# Combine and label all the sentences
count = 0
labeled_sentences = []
for sentence in sentences:
    words = sentence[0]
    count = max(count,sentence[1])
    labeled_sentence = LabeledSentence(words=words,tags=[count])
    labeled_sentences.append(labeled_sentence)
    count+=1


# Print out a sample for one to view what the structure is looking like
print "------------------ FIRST 5 SENTENCES -----------------"
pprint(labeled_sentences[:5])
print "------------------ LAST 5 SENTENCES -----------------"
pprint(labeled_sentences[-5:])
print "-----------------------------------------------------\n"


def evaluate_model(model,pairs,threshold=0.3):
    """
    Evaluate the performance of the model on a paraphrase task
    @model. The doc2vec model
    @pairs. A list of (sentence1index, sentence2index, label) tuples
    @threshold. The cosine similarity threshold for labeling paraphrases
    Returns: A new cosine similarity threshold
    """

    labels = []
    positive = []
    negative = []
    correctness = []
    predictions = []

    for tag1,tag2,label in pairs:
        docvec1 = model.docvecs[tag1].reshape(-1, 1)
        docvec2 = model.docvecs[tag2].reshape(-1, 1)
        similarity = cosine_similarity(docvec1.T, docvec2.T).flatten()
        prediction = similarity>threshold
        if label==1:
            positive.append(similarity)
        else:
            negative.append(similarity)
        labels.append(label)
        predictions.append(prediction)
        correctness.append(label==prediction)

    # Cast the types as integers
    labels = np.array(labels).astype(bool)
    predictions = np.array(predictions).astype(bool)
    # Print the summary stats for this round
    p,r,f,_ = precision_recall_fscore_support(labels, predictions, average='binary')
    print "Positive: ", np.mean(positive)
    print "Negative: ", np.mean(negative)
    print "Accuracy: ", np.mean(correctness)
    print "Precision: %.3f,  Recall:  %.3f,  F1: %.3f"%(p,r,f)
    print "\n", confusion_matrix(labels, predictions), "\n"
    print "------------------"

    # Set the new threshold as the midpoint between the predications
    threshold = np.mean(positive+negative)
    return threshold



def main():
    """
    Train the model and save it to MODEL_FILE
    Evaluate the model performance on a paraphrase data set at each step
    """

    # Mikolov pointed out that to reach a better result, you may either want to shuffle the
    # input sentences or to decrease the learning rate alpha.
    model = doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025, size=100, window=8, min_count=1, workers=8)
    model.build_vocab(labeled_sentences)

    # We start with the default threshold
    threshold = None

    for epoch in range(100):
        print "Running epoch: ",epoch
        model.train(labeled_sentences)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

        # Evaluate the model
        threshold = evaluate_model(model,pairs,threshold)

        # Shuffle for next time
        shuffle(labeled_sentences)

        # We save the model every 5 epocs
        if epoch%MODEL_SAVE_DURATION==0:
            filename = MODEL_FILE.format(epoch)
            model.save(filename)


if __name__=="__main__":
    main()


