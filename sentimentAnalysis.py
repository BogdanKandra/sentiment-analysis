"""
Created on Fri Apr 13 15:08:22 2018

@author: Bogdan
"""

import nltk
import random
import pickle
from nltk.corpus import movie_reviews, stopwords

stopwords = set(stopwords.words('english'))

# Collect the data -- movie_reviews corpus, 2000 reviews (1000 pos and 1000 neg)
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

# Randomize the data so it is suitable for training and testing sets
random.seed(69)
random.shuffle(documents)

# The Bag-of-words model
# Remove stopwords, punctuation, numerals and lowercase the rest
all_words = [w.lower() for w in movie_reviews.words() 
                if len(w) > 2 and w not in stopwords and w.isnumeric() == False]

all_words = nltk.FreqDist(all_words)

# Take most common 4000 words as features
word_features = [w[0] for w in all_words.most_common(4000)]

def find_features(document):
    """ Features (the 4000 most common words from the movie_reviews corpus)
    are searched in the document. Results are represented as a dictionary.
    """
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = w in words
        
    return features

# Find features for every review in documents
feature_sets = [(find_features(review), category) for (review, category) in documents]

training_set = feature_sets[:1900] # First 1900 reviews are for training
testing_set  = feature_sets[1900:] # Last 100 reviews are for testing

# Train a fresh classifier
#classifier = nltk.NaiveBayesClassifier.train(training_set)

# Pickle the classifier
#save_classifier = open('naivebayes.pickle', 'wb')
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

# Unpickle the trained classifier
classifier_file = open('naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_file)
classifier_file.close()

# Test
print('Naive Bayes accuracy:', nltk.classify.accuracy(classifier, testing_set) * 100)
classifier.show_most_informative_features(30)
