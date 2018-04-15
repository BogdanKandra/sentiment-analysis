"""
Created on Fri Apr 13 15:08:22 2018

@author: Bogdan
"""

import nltk
import random
import pickle
from nltk.corpus import movie_reviews, stopwords
#from nltk.classify.scikitlearn import SklearnClassifier
#from sklearn.naive_bayes import MultinomialNB, BernoulliNB
#from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)
        
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)
        
        choice_votes = votes.count(mode(votes))
        return choice_votes / len(votes)


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


# Train a fresh NLTK Naive Bayes Classifier
#classifier = nltk.NaiveBayesClassifier.train(training_set)

# Pickle the trained NB Classifier
#classifier_file = open('classifiers/nltknaivebayes.pickle', 'wb')
#pickle.dump(classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained NLTK NB Classifier
classifier_file = open('classifiers/nltknaivebayes.pickle', 'rb')
classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the NLTK NB Classifier
print('NLTK Naive Bayes accuracy:', nltk.classify.accuracy(classifier, testing_set) * 100)
#classifier.show_most_informative_features(30)

# Train a fresh Multinomial Naive Bayes Classifier
#MNB_classifier = SklearnClassifier(MultinomialNB())
#MNB_classifier.train(training_set)

# Pickle the trained MNB Classifier
#classifier_file = open('classifiers/multinomialnaivebayes.pickle', 'wb')
#pickle.dump(MNB_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained MNB classifier
classifier_file = open('classifiers/multinomialnaivebayes.pickle', 'rb')
MNB_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Multinomial NB classifier
print('Multinomial Naive Bayes accuracy:', nltk.classify.accuracy(MNB_classifier, testing_set) * 100)

# Train a fresh Bernoulli Naive Bayes Classifier
#BNB_classifier = SklearnClassifier(BernoulliNB())
#BNB_classifier.train(training_set)

# Pickle the trained BNB Classifier
#classifier_file = open('classifiers/bernoullinaivebayes.pickle', 'wb')
#pickle.dump(BNB_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained BNB classifier
classifier_file = open('classifiers/bernoullinaivebayes.pickle', 'rb')
BNB_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Bernoulli NB classifier
print('Bernoulli Naive Bayes accuracy:', nltk.classify.accuracy(BNB_classifier, testing_set) * 100)

# Train a fresh Logistic Regression Classifier
#LR_classifier = SklearnClassifier(LogisticRegression())
#LR_classifier.train(training_set)

# Pickle the trained LR Classifier
#classifier_file = open('classifiers/logisticregression.pickle', 'wb')
#pickle.dump(LR_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained LR Classifier
classifier_file = open('classifiers/logisticregression.pickle', 'rb')
LR_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Logistic Regression Classifier
print('Logistic Regression accuracy:', nltk.classify.accuracy(LR_classifier, testing_set) * 100)

# Train a fresh Stochastic Gradient Descent Classifier
#SGD_classifier = SklearnClassifier(SGDClassifier())
#SGD_classifier.train(training_set)

# Pickle the trained SGD Classifier
#classifier_file = open('classifiers/gradientdescent.pickle', 'wb')
#pickle.dump(SGD_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained SGD Classifier
classifier_file = open('classifiers/gradientdescent.pickle', 'rb')
SGD_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Stochastic Gradient Descent Classifier
print('Stochastic Gradient Descent accuracy:', nltk.classify.accuracy(SGD_classifier, testing_set) * 100)

# Train a fresh Linear Support Vector Classifier
#LSV_classifier = SklearnClassifier(LinearSVC())
#LSV_classifier.train(training_set)

# Pickle the trained LSV Classifier
#classifier_file = open('classifiers/linearsupportvector.pickle', 'wb')
#pickle.dump(LSV_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained LSV Classifier
classifier_file = open('classifiers/linearsupportvector.pickle', 'rb')
LSV_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Linear Support Vector Classifier
print('Linear Support Vectors accuracy:', nltk.classify.accuracy(LSV_classifier, testing_set) * 100)

# Train a fresh Nu Support Vector Classifier
#NuSV_classifier = SklearnClassifier(NuSVC())
#NuSV_classifier.train(training_set)

# Pickle the trained NuSV Classifier
#classifier_file = open('classifiers/nusupportvector.pickle', 'wb')
#pickle.dump(NuSV_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained NuSV Classifier
classifier_file = open('classifiers/nusupportvector.pickle', 'rb')
NuSV_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Nu Support Vector Classifier
print('Nu Support Vectors accuracy:', nltk.classify.accuracy(NuSV_classifier, testing_set) * 100)

# Create a VoteClassifier instance, which takes a decision
# based on all 8 classifiers' decisions
voted_classifier = VoteClassifier(classifier, MNB_classifier, BNB_classifier, 
                                  LR_classifier, SGD_classifier, 
                                  LSV_classifier, NuSV_classifier)

# Test the VoteClassifier
print('VoteClassifier accuracy:', nltk.classify.accuracy(voted_classifier, testing_set) * 100)

print('Classification:', voted_classifier.classify(testing_set[0][0]), 'Confidence %:', voted_classifier.confidence(testing_set[0][0]) * 100)
print('Classification:', voted_classifier.classify(testing_set[1][0]), 'Confidence %:', voted_classifier.confidence(testing_set[1][0]) * 100)
print('Classification:', voted_classifier.classify(testing_set[2][0]), 'Confidence %:', voted_classifier.confidence(testing_set[2][0]) * 100)
print('Classification:', voted_classifier.classify(testing_set[3][0]), 'Confidence %:', voted_classifier.confidence(testing_set[3][0]) * 100)
print('Classification:', voted_classifier.classify(testing_set[4][0]), 'Confidence %:', voted_classifier.confidence(testing_set[4][0]) * 100)
