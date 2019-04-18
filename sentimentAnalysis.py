"""
Created on Fri Apr 13 15:08:22 2018

@author: Bogdan
"""
import nltk, pickle, random
import proj_utils as utils
from nltk.classify.scikitlearn import SklearnClassifier

# Trains a classifier, serializes the result and tests it
def trainPickleTestClassifier(estimator):
    # Train the classifier
    classifier = SklearnClassifier(estimator)
    classifier.train(training_set)
    
    # Serialize the trained classifier
    path = 'pickles/classifiers/' + estimator.__class__.__name__ + '.pickle'
    file = open(path, 'wb')
    pickle.dump(classifier, file)
    file.close()
    
    # Test the classifier
    print(estimator.__class__.__name__ + ' accuracy:', 
          nltk.classify.accuracy(classifier, testing_set) * 100)
    utils.print_precision_recall(classifier, testing_set)


# Collect the data and serialize it -- movie_reviews corpus, 2000 reviews (1000 pos and 1000 neg)
documents = utils.get_documents()
random.shuffle(documents)  # Randomize the list of documents
file = open('pickles/data/documents.pickle', 'wb')
pickle.dump(documents, file)
file.close()

# Bag-of-words Model
all_words = utils.get_words()
all_words = nltk.FreqDist(all_words)

# Consider the most common 5000 words as features
word_features = [w[0] for w in all_words.most_common(5000)]
file = open('pickles/data/wordfeatures.pickle', 'wb')
pickle.dump(word_features, file)
file.close()

# Find features for every review in documents
feature_sets = [(utils.get_features(review, word_features), category) for (review, category) in documents]
file = open('pickles/data/featuresets.pickle', 'wb')
pickle.dump(feature_sets, file)
file.close()

# Unpickle the documents and features
#f = open('pickles/data/documents.pickle', 'rb')
#documents = pickle.load(f)
#f.close()
#
#f = open('pickles/data/wordfeatures.pickle', 'rb')
#word_features = pickle.load(f)
#f.close()
#
#f = open('pickles/data/featuresets.pickle', 'rb')
#feature_sets = pickle.load(f)
#f.close()

# Select training and testing sets
# First 900 positive and negative are for training
training_set = feature_sets[0:900] + feature_sets[1000:1900]
testing_set  = [elem for elem in feature_sets if elem not in training_set]

# Start the training process for the selected estimators
estimators = utils.ESTIMATORS

for estimator in estimators:
    trainPickleTestClassifier(estimator)
