#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import sys

import numpy as np

PARENT_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..')
sys.path.insert(0, PARENT_DIR)

from falsefriends import classifier, linear_trans, util, word_vectors

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Method\t Acc")

friend_pairs = util.read_words(PARENT_DIR + '/resources/sepulveda2011_original.txt')

VECTOR_SIZES = [100, 200, 400, 800]

for size_es in VECTOR_SIZES:
    model_es = word_vectors.load_model(PARENT_DIR + '/resources/big/vectors_es_{}.bin'.format(size_es))
    for size_pt in VECTOR_SIZES:
        model_pt = word_vectors.load_model(PARENT_DIR + '/resources/big/vectors_pt_{}.bin'.format(size_pt))

        clf = classifier.build_classifier()

        T = linear_trans.load_linear_transformation(PARENT_DIR + '/resources/big/trans_es_{}_pt_{}.npz'.format(size_es,
                                                                                                               size_pt))

        X, y = classifier.features_and_labels(friend_pairs, model_es, model_pt, T)

        measures = classifier.classify_with_cross_validation(X, y, clf=clf)

        print("{}\t{:0.4f}".format("es-{}-{}-1".format(size_es, size_pt), measures['Accuracy'][0]))

        clf = classifier.build_classifier()

        T = linear_trans.load_linear_transformation(PARENT_DIR + '/resources/big/trans_pt_{}_es_{}.npz'.format(size_pt,
                                                                                                               size_es))

        X, y = classifier.features_and_labels(friend_pairs, model_es, model_pt, T, backwards=True)

        measures = classifier.classify_with_cross_validation(X, y, clf=clf)

        print("{}\t{:0.4f}".format("pt-{}-{}-1".format(size_pt, size_es), measures['Accuracy'][0]))
