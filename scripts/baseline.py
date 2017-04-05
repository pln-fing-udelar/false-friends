#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus import wordnet as wn
import numpy as np

from falsefriends import util

testing_friend_pairs = util.read_words('resources/sepulveda2011_training.txt')

expected = np.asarray([friend_pair.true_friends for friend_pair in testing_friend_pairs])

outcome = np.asarray([any(friend_pair.word_pt in synset.lemma_names(lang='por')
                          for synset in wn.synsets(friend_pair.word_es, lang='spa'))
                      for friend_pair in testing_friend_pairs])

print('Accuracy: {}'.format(np.mean(expected == outcome)))
