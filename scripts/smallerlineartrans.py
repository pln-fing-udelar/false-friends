#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import logging
from tqdm import *

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from falsefriends import linear_trans, word_vectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_es = word_vectors.load_model('../resources/big/jairo/vectors_es.bin')
model_pt = word_vectors.load_model('../resources/big/jairo/vectors_pt.bin')

logging.info("getting vector pairs")
X, Y = zip(*word_vectors.bilingual_lexicon_vectors(model_es, model_pt))

X_array = np.vstack(X)
Y_array = np.vstack(Y)

sizes = [50, 100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, len(X)]
for size in tqdm(sizes):
    indices = np.random.choice(list(range(X_array.shape[0])), size, replace=False)

    T = linear_trans.linear_transformation(list(X_array[indices, :]), list(Y_array[indices, :]))
    linear_trans.save_linear_transformation('../resources/big/jairo/linear_trans_{}'.format(size), T)
