"""
Created on Tue May  1 21:19:33 2018

@author: Bogdan
"""
import collections
from nltk import precision, recall
from nltk.corpus import stopwords, movie_reviews
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))

def get_documents():
    
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((list(movie_reviews.words(fileid)), category))
            
    return documents

def get_words():
    "Remove stopwords, punctuation, numerals and lowercase the rest"
    
    words = [w.lower() for w in movie_reviews.words() 
                if len(w) > 2 and w not in STOPWORDS and w.isnumeric() == False]
    
    return words

def get_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def print_precision_recall(classifier, test_set):
    
    # Build dicts containing known and computed labels for the test data
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

    # Build dicts containing known and computed labels for the test data
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

