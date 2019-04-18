"""
Created on Thu May  3 21:46:14 2018

@author: Bogdan
"""
import pickle
import proj_utils as utils
from voteClassifier import VoteClassifier

# Unserializes a trained classifier and returns it
def get_classifier(estimator):
    path = 'pickles/classifiers/' + estimator.__class__.__name__ + '.pickle'
    f = open(path, 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier

# Predicts the sentiment of a text on demand, using the Voting classifier
def sentiment(text):
    features = utils.get_text_features(text, word_features)
    return voted_classifier.classify(features), voted_classifier.confidence(features)


# Deserialize the word features
f = open('pickles/data/wordfeatures.pickle', 'rb')
word_features = pickle.load(f)
f.close()

# Create a VoteClassifier instance by passing the trained classifiers list
# It takes a decision based on all classifiers' decisions
trainedClassifiers = [get_classifier(estimator) for estimator in utils.ESTIMATORS]
voted_classifier = VoteClassifier(trainedClassifiers)

# Predict texts
text1 = "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"
text2 = "I think this movie was extremely good, fabulous, superb, smart and beautiful, great."
text3 = 'beautiful good great amazing wonderful'
text4 = "If you are a preteen, the story may work for you. If you have any common sense, the bad dialogue and \
        blatant rip off of every sci-fi and fantasy movie cliche will drive you crazy. Special effects are decent, \
        and the story moves along at a decent pace. Momoa carries the film despite being surrounded by one \
        dimensional characters with digitally botoxed faces. I would never watch this movie again."
text5 = "I don't know where the hype for this film comes from, because it can't be from the content. \
        The dialogue and characters in this movie are often ridiculous in the least entertaining way possible. \
        Silly can be entertaining, but it can also just feel stupid. The female characters only seem to exist to \
        be Aquaman's conscience (because he can't think for himself, as he's boring), they have no flaws or vices \
        (in other words no depth). The villain is a cheap charicature of the warmonger, with a few poorly expressed \
        environmental ideas thrown into his perspective. The good points of this film fall mostly on the visuals. \
        It is a stunning film in terms of the undersea beauty portrayed, a literal riot of colour across the screen. \
        I would also say that action is pretty positive, the combat is intense and epic. Overral though, I wouldn't reccomend"
text6 = "I have no idea why somebody would give this movie a good score, let alone 10. After 20 minutes I have decided that \
        its for the best to switch my brain off and enjoy the show. But damn this movie is way too dumb. Character development \
        is rushed and is given in a form of cringy and weak dialogues. Jokes feel like they are just out of place, but specific \
        moments are created just for them to happen (which makes it even more ridiculous). There are some really stunning shots, \
        but its more of an exception rather than the rule. But the main issue, which makes this movie unwatchable without at least \
        4 cans of beer, is the fact that its linear as hell, there are 0 unpredictable plot twists, from the beginning of the movie \
        (or even from trailers) you for sure know how its going to end. Also the mood is unbelievably inconsistent, at some points \
        characters are dead serious, while in 2 minutes they may be cracking jokes. Its not as bad as in Justice League, but black \
        manta just looks pathetic with his constanty frowny or angry face. WB, u evil corporate idiots, stop ruining your movies by \
        trying to be like Marvel and making movies based on a formula which clearly doesn't work. It just looks like a parody and not a well done one."
text7 = "Watched it!!! And have no words to describe the splendid performance of the cast and how beautifully wan visualised and presented \
        this..This is the true jewel so far in DC universe👍🏻 Won't Add any Spoilers😋..as this is something you should witness"

print('Classification:', sentiment(text7))
