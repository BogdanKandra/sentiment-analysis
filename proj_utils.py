"""
Created on Tue May  1 21:19:33 2018

@author: Bogdan
"""
import collections
from nltk import precision, recall
from nltk.corpus import stopwords, movie_reviews
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


STOPWORDS = set(stopwords.words('english'))

ESTIMATORS = [
        BernoulliNB(),
        MultinomialNB(),
        LogisticRegression(),
        LogisticRegressionCV(),
        SGDClassifier(),
        LinearSVC(),
        NuSVC(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        MLPClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier()
        ]

def get_documents():
    """ Builds the documents set as a list of pairs representing the list of 
    words present in a review and its category (positive or negative). 
    The source of the documents is the movie reviews dataset
    """
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((list(movie_reviews.words(fileid)), category))
    
    return documents

def get_words():
    """ Returns a list containing all words from the movie reviews dataset.
    The words are preprocessed by removing stopwords, numerals and punctuation, and lowercasing
    """
    words = [w.lower() for w in movie_reviews.words() 
                if len(w) > 2 and w not in STOPWORDS and w.isnumeric() == False]
    
    return words

def get_word_list_features(word_list, word_features):
    """ Builds a features dictionary from a list of words.
    Used for obtaining the features from the dataset reviews
    """
    document = ' '.join(word_list)
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def get_text_features(text, word_features):
    """ Builds a features dictionary from a string.
    Used for obtaining the features from any text
    """
    words = word_tokenize(text)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def print_precision_recall(classifier, test_set):
    """ Computes and prints the precision and recall metrics, given a classifier
    and a test set
    """
    known_set = collections.defaultdict(set)
    computed_set = collections.defaultdict(set)
    
    for i, (features, label) in enumerate(test_set):
        known_set[label].add(i)
        predicted = classifier.classify(features)
        computed_set[predicted].add(i)

    print('pos precision:', precision(known_set['pos'], computed_set['pos']))
    print('pos recall:', recall(known_set['pos'], computed_set['pos']))
    print('neg precision:', precision(known_set['neg'], computed_set['neg']))
    print('neg recall:', recall(known_set['neg'], computed_set['neg']))

def log_precision_recall(classifier, test_set, file):
    """ Computes and logs the precision and recall metrics into a file, given a 
    classifier and a test set
    """
    known_set = collections.defaultdict(set)
    computed_set = collections.defaultdict(set)
    
    for i, (features, label) in enumerate(test_set):
        known_set[label].add(i)
        predicted = classifier.classify(features)
        computed_set[predicted].add(i)

    line = 'pos precision: ' + str(precision(known_set['pos'], computed_set['pos']))
    file.write(line + '\n')
    line = 'pos recall: ' + str(recall(known_set['pos'], computed_set['pos']))
    file.write(line + '\n')
    line = 'neg precision: ' + str(precision(known_set['neg'], computed_set['neg']))
    file.write(line + '\n')
    line = 'neg recall: ' + str(recall(known_set['neg'], computed_set['neg']))
    file.write(line + '\n')
