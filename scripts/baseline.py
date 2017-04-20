#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus import wordnet as wn
import numpy as np
import os
import sys

PARENT_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..')
sys.path.insert(0, PARENT_DIR)

from falsefriends import util

# As we are comparing methods with cross-validation and this method doesn't fit parameters,
# we evaluate with the whole dataset.
friend_pairs = util.read_words(PARENT_DIR + '/resources/sepulveda2011_original.txt')

covered_pairs = [friend_pair for friend_pair in friend_pairs
                 if friend_pair.word_es in wn.all_lemma_names(lang='spa')
                 and friend_pair.word_pt in wn.all_lemma_names(lang='por')]

expected = np.asarray([friend_pair.true_friends for friend_pair in covered_pairs])

outcome = np.asarray([any(friend_pair.word_pt in synset.lemma_names(lang='por')
                          for synset in wn.synsets(friend_pair.word_es, lang='spa'))
                      for friend_pair in covered_pairs])

print('Accuracy: {}'.format(np.mean(expected == outcome)))
print('Coverage: {}'.format(len(covered_pairs) / len(friend_pairs)))
