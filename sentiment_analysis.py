"""
Created on Thu May  3 21:46:14 2018

@author: Bogdan
"""
import pickle
import proj_utils as utils
from voteClassifier import VoteClassifier

def analyse(text):
    feats = utils.get_features(text, word_features)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

# Unpickle the documents and features
f = open('pickles/twitter_vars/documents.pickle', 'rb')
documents = pickle.load(f)
f.close()

f = open('pickles/twitter_vars/wordfeatures.pickle', 'rb')
word_features = pickle.load(f)
f.close()

f = open('pickles/twitter_vars/featuresets.pickle', 'rb')
feature_sets = pickle.load(f)
f.close()

# Unpickle the trained classifiers
f = open('pickles/twitter_classifiers/nltknaivebayes.pickle', 'rb')
NB_classifier = pickle.load(f)
f.close()

f = open('pickles/twitter_classifiers/multinomialnaivebayes.pickle', 'rb')
MNB_classifier = pickle.load(f)
f.close()

f = open('pickles/twitter_classifiers/bernoullinaivebayes.pickle', 'rb')
BNB_classifier = pickle.load(f)
f.close()

f = open('pickles/twitter_classifiers/logisticregression.pickle', 'rb')
LR_classifier = pickle.load(f)
f.close()

f = open('pickles/twitter_classifiers/gradientdescent.pickle', 'rb')
SGD_classifier = pickle.load(f)
f.close()

f = open('pickles/twitter_classifiers/linearsupportvector.pickle', 'rb')
LSV_classifier = pickle.load(f)
f.close()

f = open('pickles/twitter_classifiers/nusupportvector.pickle', 'rb')
NuSV_classifier = pickle.load(f)
f.close()

voted_classifier = VoteClassifier(NB_classifier, MNB_classifier, BNB_classifier, 
                                  LR_classifier, SGD_classifier, 
                                  LSV_classifier, NuSV_classifier)


#text1 = "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"
#text2 = "I think this movie was extremely good, fabulous, superb, smart and beautiful, great."
#text3 = "This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"

#print('Classification:', analyse(text1))
