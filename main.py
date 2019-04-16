"""
Created on Thu May  3 21:46:14 2018

@author: Bogdan
"""
import pickle
import proj_utils as utils
from voteClassifier import VoteClassifier

# Predicts the sentiment of a text on demand
def sentiment(text):
    features = utils.find_features(text, word_features)
    return voted_classifier.classify(features), voted_classifier.confidence(features)

# Unpickle the documents and features
f = open('pickles/data/documents.pickle', 'rb')
documents = pickle.load(f)
f.close()

f = open('pickles/data/wordfeatures.pickle', 'rb')
word_features = pickle.load(f)
f.close()

f = open('pickles/data/featuresets.pickle', 'rb')
feature_sets = pickle.load(f)
f.close()

# Unpickle the trained classifiers
f = open('pickles/classifiers/nltknaivebayes.pickle', 'rb')
NB_classifier = pickle.load(f)
f.close()

f = open('pickles/classifiers/multinomialnaivebayes.pickle', 'rb')
MNB_classifier = pickle.load(f)
f.close()

f = open('pickles/classifiers/bernoullinaivebayes.pickle', 'rb')
BNB_classifier = pickle.load(f)
f.close()

f = open('pickles/classifiers/logisticregression.pickle', 'rb')
LR_classifier = pickle.load(f)
f.close()

f = open('pickles/classifiers/gradientdescent.pickle', 'rb')
SGD_classifier = pickle.load(f)
f.close()

f = open('pickles/classifiers/linearsupportvector.pickle', 'rb')
LSV_classifier = pickle.load(f)
f.close()

f = open('pickles/classifiers/nusupportvector.pickle', 'rb')
NuSV_classifier = pickle.load(f)
f.close()

voted_classifier = VoteClassifier(NB_classifier, MNB_classifier, BNB_classifier, 
                                  LR_classifier, SGD_classifier, 
                                  LSV_classifier, NuSV_classifier)


text1 = "I work at a movie theater and every Thursday night we have an employee screening of one movie that comes out the next day...Today it was The Guardian. I saw the trailers and the ads and never expected much from it, and in no way really did i anticipate seeing this movie. Well turns out this movie was a lot more than I would have thought. It was a great story first of all. Ashton Kutcher and Kevin Costner did amazing acting work in this film. Being a big fan of That 70's Show I always found it hard thinking of Kutcher as anyone but Kelso despite the great acting he did in The Butterfly Effect, but after seeing this movie I think I might be able to finally look at him as a serious actor.<br /><br />It was also a great tribute to the unsung heroes of the U.S. Coast Guard."
text2 = "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"
text3 = "I think this movie was extremely good, fabulous, superb, smart and beautiful, great."
text4 = "This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"

#print('Classification:', analyse(text1))
