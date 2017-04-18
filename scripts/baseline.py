#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus import wordnet as wn
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from falsefriends import util

testing_friend_pairs = util.read_words('../resources/sepulveda2011_testing.txt')

covered_pairs = [friend_pair for friend_pair in testing_friend_pairs
                 if friend_pair.word_es in wn.all_lemma_names(lang='spa')
                 and friend_pair.word_pt in wn.all_lemma_names(lang='por')]

expected = np.asarray([friend_pair.true_friends for friend_pair in covered_pairs])

outcome = np.asarray([any(friend_pair.word_pt in synset.lemma_names(lang='por')
                          for synset in wn.synsets(friend_pair.word_es, lang='spa'))
                      for friend_pair in covered_pairs])

print('Accuracy: {}'.format(np.mean(expected == outcome)))
print('Coverage: {}'.format(len(covered_pairs) / len(testing_friend_pairs)))
