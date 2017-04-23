#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import random

from sklearn import svm

from falsefriends import classifier
from falsefriends import linear_trans
from falsefriends import util, word_vectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

training_friend_pairs = util.read_words('resources/sepulveda2011_training.txt')

model_es = word_vectors.load_model('resources/big/vectors_es_100.bin')
model_pt = word_vectors.load_model('resources/big/vectors_pt_100.bin')

logging.info("computing equal words...")
equal_words = model_es.vocab.keys() & model_pt.vocab.keys()

print("Equal words number in the Wikipedia's:", len(equal_words))

SAMPLE_SIZE = 20
print("Sample", SAMPLE_SIZE, "equal words found:", random.sample(equal_words, SAMPLE_SIZE))

T = linear_trans.load_linear_transformation('resources/big/trans_es_100_pt_100.npz')

clf = classifier.build_classifier()

X_train, y_train = classifier.features_and_labels(training_friend_pairs, model_es, model_pt, T)

logging.info("training...")
clf.fit(X_train, y_train)

equal_friend_pairs = (classifier.FriendPair(word, word, None) for word in equal_words)

logging.info("computing features...")
X_equal, _ = classifier.features_and_labels(equal_friend_pairs, model_es, model_pt, T)

logging.info("predicting equal words...")
y_equal = clf.predict(X_equal)

print("Cognates percentage in equal words in Wikipedia's:", sum(y_equal) / len(y_equal))

# TODO: same for similar (but not equal) words
