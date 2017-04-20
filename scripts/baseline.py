#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus import wordnet as wn
import numpy as np

from falsefriends import util

testing_friend_pairs = util.read_words('../resources/sepulveda2011_testing.txt')

words_covered = list(filter(lambda friend_pair: len(wn.synsets(friend_pair.word_es, lang='spa')) > 0
                                           and len(wn.synsets(friend_pair.word_pt, lang='por')) > 0
                            , testing_friend_pairs))
expected = np.asarray([friend_pair.true_friends for friend_pair in words_covered])

outcome = np.asarray([any(friend_pair.word_pt in synset.lemma_names(lang='por')
                          for synset in wn.synsets(friend_pair.word_es, lang='spa'))
                      for friend_pair in words_covered])

print('Accuracy: {}'.format(np.mean(expected == outcome)))
print('Coverage: {}'.format(len(words_covered)/float(len(testing_friend_pairs))))
