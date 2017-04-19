#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import logging
from tqdm import *

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from falsefriends import bilingual_lexicon, linear_trans, word_vectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_es = word_vectors.load_model('../resources/big/jairo/vectors_es.bin')
model_pt = word_vectors.load_model('../resources/big/jairo/vectors_pt.bin')

lexicon = bilingual_lexicon.most_frequent_bilingual_lexicon_based_on_external_count(model_es.vocab, model_pt.vocab)

logging.info("getting vector pairs")
X, Y = zip(*word_vectors.bilingual_lexicon_vectors(model_es, model_pt, bilingual_lexicon=lexicon))

X_array = np.vstack(X)
Y_array = np.vstack(Y)

sizes = [50, 100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, len(X)]
sizes = [size for size in sizes if size <= len(X)]
for size in tqdm(sizes):
    indices = np.random.choice(list(range(X_array.shape[0])), size, replace=False)

    T = linear_trans.linear_transformation(list(X_array[indices, :]), list(Y_array[indices, :]))
    linear_trans.save_linear_transformation('../resources/big/jairo/linear_trans_most_frequent_{}'.format(size), T)
