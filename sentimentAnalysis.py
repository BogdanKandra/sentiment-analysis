"""
Created on Fri Apr 13 15:08:22 2018

@author: Bogdan
"""
import nltk, pickle
import proj_utils as utils
from voteClassifier import VoteClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# Collect the data -- movie_reviews corpus, 2000 reviews (1000 pos and 1000 neg)
#documents = utils.get_documents()
#file = open('pickles/vars/documents.pickle', 'wb')
#pickle.dump(documents, file)
#file.close()

# Unpickle the documents variable
file = open('pickles/vars/documents.pickle', 'rb')
documents = pickle.load(file)
file.close()

# Bag-of-words Model
#all_words = utils.get_words()
#all_words = nltk.FreqDist(all_words)
#
## Consider the most common 5000 words as features
#word_features = [w[0] for w in all_words.most_common(5000)]
#file = open('pickles/vars/wordfeatures.pickle', 'wb')
#pickle.dump(word_features, file)
#file.close()

# Unpickle the word features
file = open('pickles/vars/wordfeatures.pickle', 'rb')
word_features = pickle.load(file)
file.close()

# Find features for every review in documents
feature_sets = [(utils.find_features(review, word_features), category) for (review, category) in documents]

# Select training and testing sets
# First 900 positive and negative are for training
training_set = feature_sets[0:900] + feature_sets[1000:1900]
testing_set  = [elem for elem in feature_sets if elem not in training_set]

# Train a fresh NLTK Naive Bayes Classifier
#classifier = nltk.NaiveBayesClassifier.train(training_set)

# Pickle the trained NB Classifier
#classifier_file = open('pickles/classifiers/nltknaivebayes.pickle', 'wb')
#pickle.dump(classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained NLTK NB Classifier
classifier_file = open('pickles/classifiers/nltknaivebayes.pickle', 'rb')
classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the NLTK NB Classifier
#print('NLTK Naive Bayes accuracy:', nltk.classify.accuracy(classifier, testing_set) * 100)
#utils.print_precision_recall(classifier, testing_set)
#classifier.show_most_informative_features(10)

# Train a fresh Multinomial Naive Bayes Classifier
#MNB_classifier = SklearnClassifier(MultinomialNB())
#MNB_classifier.train(training_set)

# Pickle the trained MNB Classifier
#classifier_file = open('pickles/classifiers/multinomialnaivebayes.pickle', 'wb')
#pickle.dump(MNB_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained MNB classifier
classifier_file = open('pickles/classifiers/multinomialnaivebayes.pickle', 'rb')
MNB_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Multinomial NB classifier
#print('Multinomial Naive Bayes accuracy:', nltk.classify.accuracy(MNB_classifier, testing_set) * 100)
#utils.print_precision_recall(MNB_classifier, testing_set)

# Train a fresh Bernoulli Naive Bayes Classifier
#BNB_classifier = SklearnClassifier(BernoulliNB())
#BNB_classifier.train(training_set)

# Pickle the trained BNB Classifier
#classifier_file = open('pickles/classifiers/bernoullinaivebayes.pickle', 'wb')
#pickle.dump(BNB_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained BNB classifier
classifier_file = open('pickles/classifiers/bernoullinaivebayes.pickle', 'rb')
BNB_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Bernoulli NB classifier
#print('Bernoulli Naive Bayes accuracy:', nltk.classify.accuracy(BNB_classifier, testing_set) * 100)
#utils.print_precision_recall(BNB_classifier, testing_set)

# Train a fresh Logistic Regression Classifier
#LR_classifier = SklearnClassifier(LogisticRegression())
#LR_classifier.train(training_set)

# Pickle the trained LR Classifier
#classifier_file = open('pickles/classifiers/logisticregression.pickle', 'wb')
#pickle.dump(LR_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained LR Classifier
classifier_file = open('pickles/classifiers/logisticregression.pickle', 'rb')
LR_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Logistic Regression Classifier
#print('Logistic Regression accuracy:', nltk.classify.accuracy(LR_classifier, testing_set) * 100)
#utils.print_precision_recall(LR_classifier, testing_set)

# Train a fresh Stochastic Gradient Descent Classifier
#SGD_classifier = SklearnClassifier(SGDClassifier())
#SGD_classifier.train(training_set)

# Pickle the trained SGD Classifier
#classifier_file = open('pickles/classifiers/gradientdescent.pickle', 'wb')
#pickle.dump(SGD_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained SGD Classifier
classifier_file = open('pickles/classifiers/gradientdescent.pickle', 'rb')
SGD_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Stochastic Gradient Descent Classifier
#print('Stochastic Gradient Descent accuracy:', nltk.classify.accuracy(SGD_classifier, testing_set) * 100)
#utils.print_precision_recall(SGD_classifier, testing_set)

# Train a fresh Linear Support Vector Classifier
#LSV_classifier = SklearnClassifier(LinearSVC())
#LSV_classifier.train(training_set)

# Pickle the trained LSV Classifier
#classifier_file = open('pickles/classifiers/linearsupportvector.pickle', 'wb')
#pickle.dump(LSV_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained LSV Classifier
classifier_file = open('pickles/classifiers/linearsupportvector.pickle', 'rb')
LSV_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Linear Support Vector Classifier
#print('Linear Support Vectors accuracy:', nltk.classify.accuracy(LSV_classifier, testing_set) * 100)
#utils.print_precision_recall(LSV_classifier, testing_set)

# Train a fresh Nu Support Vector Classifier
#NuSV_classifier = SklearnClassifier(NuSVC())
#NuSV_classifier.train(training_set)

# Pickle the trained NuSV Classifier
#classifier_file = open('pickles/classifiers/nusupportvector.pickle', 'wb')
#pickle.dump(NuSV_classifier, classifier_file)
#classifier_file.close()

# Unpickle the trained NuSV Classifier
classifier_file = open('pickles/classifiers/nusupportvector.pickle', 'rb')
NuSV_classifier = pickle.load(classifier_file)
classifier_file.close()

# Test the Nu Support Vector Classifier
#print('Nu Support Vectors accuracy:', nltk.classify.accuracy(NuSV_classifier, testing_set) * 100)
#utils.print_precision_recall(NuSV_classifier, testing_set)

# Create a VoteClassifier instance, which takes a decision
# based on all 7 classifiers' decisions
voted_classifier = VoteClassifier(classifier, MNB_classifier, BNB_classifier, 
                                  LR_classifier, SGD_classifier, 
                                  LSV_classifier, NuSV_classifier)

# Test the VoteClassifier
#print('VoteClassifier accuracy:', nltk.classify.accuracy(voted_classifier, testing_set) * 100)
#utils.print_precision_recall(voted_classifier, testing_set)

#print('Classification:', voted_classifier.classify(testing_set[0][0]), 'Confidence %:', voted_classifier.confidence(testing_set[0][0]) * 100)

def sentiment(text):
    features = utils.find_features(text, word_features)
    return voted_classifier.classify(features)

def confidence(text):
    features = utils.find_features(text, word_features)
    return voted_classifier.confidence(features)


#text1 = "I work at a movie theater and every Thursday night we have an employee screening of one movie that comes out the next day...Today it was The Guardian. I saw the trailers and the ads and never expected much from it, and in no way really did i anticipate seeing this movie. Well turns out this movie was a lot more than I would have thought. It was a great story first of all. Ashton Kutcher and Kevin Costner did amazing acting work in this film. Being a big fan of That 70's Show I always found it hard thinking of Kutcher as anyone but Kelso despite the great acting he did in The Butterfly Effect, but after seeing this movie I think I might be able to finally look at him as a serious actor.<br /><br />It was also a great tribute to the unsung heroes of the U.S. Coast Guard."
text2 = "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"
text3 = "I think this movie was extremely good, fabulous, superb, smart and beautiful, great."
text4 = "This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"

print('Classification:', sentiment(text3), 'Confidence:', confidence(text3))
#print('Classification:', sentiment(text2), 'Confidence:', confidence(text2))
