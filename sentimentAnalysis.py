"""
Created on Fri Apr 13 15:08:22 2018

@author: Bogdan
"""

import nltk
import random
from nltk.corpus import movie_reviews

# Build the data
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

# The Bag-of-words model
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(30))


### Workflow #############
# Data preprocessing - Lowercase, Tokenize, Stopwords, Stem
# Feature extraction
# Training
# Testing