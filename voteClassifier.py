"""
Created on Tue May  1 23:10:45 2018

@author: Bogdan
"""
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, classifierList):
        self._classifiers = classifierList
    
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
