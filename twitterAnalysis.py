"""
Created on Sun Apr 15 20:45:03 2018

@author: Bogdan
"""
import nltk, pickle
import proj_utils as utils
from voteClassifier import VoteClassifier
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

def analyse(text):
    feats = utils.get_features(text, word_features)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

# Collect the data -- 10664 reviews, half positive and half negative
short_pos = open('short_reviews/positive.txt', 'r').read()
short_neg = open('short_reviews/negative.txt', 'r').read()

all_words = []
documents = []

# Filter out words which are not Adjectives
allowed_pos = ['J', 'R', 'V']

for sent in short_pos.split('\n'):
    documents.append((sent, 'pos'))
    words = word_tokenize(sent)
    tags = nltk.pos_tag(words)
    for tag in tags:
        if tag[1][0] in allowed_pos and len(tag[0]) > 2  \
            and tag[0] not in utils.STOPWORDS and tag[0].isnumeric() == False:
            all_words.append(tag[0].lower())

for sent in short_neg.split('\n'):
    documents.append( (sent, 'neg') )
    words = word_tokenize(sent)
    tags = nltk.pos_tag(words)
    for tag in tags:
        if tag[1][0] in allowed_pos and len(tag[0]) > 2  \
            and tag[0] not in utils.STOPWORDS and tag[0].isnumeric() == False:
            all_words.append(tag[0].lower())

# Pickle all the variables
f = open('pickles/twitter_vars/documents.pickle', 'wb')
pickle.dump(documents, f)
f.close()

# Features are the 5000 most common words found in the reviews
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

f = open('pickles/twitter_vars/wordfeatures.pickle', 'wb')
pickle.dump(word_features, f)
f.close()

# Build the feature sets
feature_sets = [(utils.get_features(rev, word_features), category) for (rev, category) in documents]

f = open('pickles/twitter_vars/featuresets.pickle', 'wb')
pickle.dump(feature_sets, f)
f.close()

# Select training and testing sets
# First 5000 positive and negative reviews are for training
training_set = feature_sets[0:5000] + feature_sets[5332:10332]
testing_set  = [elem for elem in feature_sets if elem not in training_set]

log_file = open('logs/before3.txt', 'a')

# Train a fresh NLTK Naive Bayes Classifier
NB_classifier = nltk.NaiveBayesClassifier.train(training_set)

# Pickle the trained NB Classifier
f = open('pickles/twitter_classifiers/nltknaivebayes.pickle', 'wb')
pickle.dump(NB_classifier, f)
f.close()

# Test the NLTK NB Classifier
line_write = 'Naive Bayes accuracy: ' + str(nltk.classify.accuracy(NB_classifier, testing_set) * 100)
log_file.write(line_write + '\n')
utils.log_precision_recall(NB_classifier, testing_set, log_file)
NB_classifier.show_most_informative_features(15)

# Train a fresh Multinomial Naive Bayes Classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

# Pickle the trained MNB Classifier
f = open('pickles/twitter_classifiers/multinomialnaivebayes.pickle', 'wb')
pickle.dump(MNB_classifier, f)
f.close()

# Test the Multinomial NB classifier
line_write = 'Multinomial Naive Bayes accuracy: ' + str(nltk.classify.accuracy(MNB_classifier, testing_set) * 100)
log_file.write(line_write + '\n')
utils.log_precision_recall(MNB_classifier, testing_set, log_file)

# Train a fresh Bernoulli Naive Bayes Classifier
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)

# Pickle the trained BNB Classifier
f = open('pickles/twitter_classifiers/bernoullinaivebayes.pickle', 'wb')
pickle.dump(BNB_classifier, f)
f.close()

# Test the Bernoulli NB classifier
line_write = 'Bernoulli Naive Bayes accuracy: ' + str(nltk.classify.accuracy(BNB_classifier, testing_set) * 100)
log_file.write(line_write + '\n')
utils.log_precision_recall(BNB_classifier, testing_set, log_file)

# Train a fresh Logistic Regression Classifier
LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)

# Pickle the trained LR Classifier
f = open('pickles/twitter_classifiers/logisticregression.pickle', 'wb')
pickle.dump(LR_classifier, f)
f.close()

# Test the Logistic Regression Classifier
line_write = 'Logistic Regression accuracy: ' + str(nltk.classify.accuracy(LR_classifier, testing_set) * 100)
log_file.write(line_write + '\n')
utils.log_precision_recall(LR_classifier, testing_set, log_file)

# Train a fresh Stochastic Gradient Descent Classifier
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)

# Pickle the trained SGD Classifier
f = open('pickles/twitter_classifiers/gradientdescent.pickle', 'wb')
pickle.dump(SGD_classifier, f)
f.close()

# Test the Stochastic Gradient Descent Classifier
line_write = 'Stochastic Gradient Descent accuracy: ' + str(nltk.classify.accuracy(SGD_classifier, testing_set) * 100)
log_file.write(line_write + '\n')
utils.log_precision_recall(SGD_classifier, testing_set, log_file)

# Train a fresh Linear Support Vector Classifier
LSV_classifier = SklearnClassifier(LinearSVC())
LSV_classifier.train(training_set)

# Pickle the trained LSV Classifier
f = open('pickles/twitter_classifiers/linearsupportvector.pickle', 'wb')
pickle.dump(LSV_classifier, f)
f.close()

# Test the Linear Support Vector Classifier
line_write = 'Linear Support Vectors accuracy: ' + str(nltk.classify.accuracy(LSV_classifier, testing_set) * 100)
log_file.write(line_write + '\n')
utils.log_precision_recall(LSV_classifier, testing_set, log_file)

# Train a fresh Nu Support Vector Classifier
NuSV_classifier = SklearnClassifier(NuSVC())
NuSV_classifier.train(training_set)

# Pickle the trained NuSV Classifier
f = open('pickles/twitter_classifiers/nusupportvector.pickle', 'wb')
pickle.dump(NuSV_classifier, f)
f.close()

# Test the Nu Support Vector Classifier
line_write = 'Nu Support Vectors accuracy: ' + str(nltk.classify.accuracy(NuSV_classifier, testing_set) * 100)
log_file.write(line_write + '\n')
utils.log_precision_recall(NuSV_classifier, testing_set, log_file)

# Create a VoteClassifier instance, which takes a decision
# based on all 7 classifiers' decisions
voted_classifier = VoteClassifier(NB_classifier, MNB_classifier, BNB_classifier, 
                                  LR_classifier, SGD_classifier, 
                                  LSV_classifier, NuSV_classifier)

# Test the VoteClassifier
line_write = 'VoteClassifier accuracy: ' + str(nltk.classify.accuracy(voted_classifier, testing_set) * 100)
log_file.write(line_write + '\n')
utils.log_precision_recall(voted_classifier, testing_set, log_file)

log_file.close()

#print('Classification:', voted_classifier.classify(testing_set[0][0]), 'Confidence %:', voted_classifier.confidence(testing_set[0][0]) * 100)
