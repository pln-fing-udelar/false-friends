#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import sys

import numpy as np

PARENT_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..')
sys.path.insert(0, PARENT_DIR)

from falsefriends import bilingual_lexicon, classifier, linear_trans, util, word_vectors

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Method\t\t Acc")

friend_pairs = util.read_words(PARENT_DIR + '/resources/sepulveda2011_original.txt')

VECTOR_SIZES = [100, 200, 400, 800]

for size_es in VECTOR_SIZES:
    model_es = word_vectors.load_model(PARENT_DIR + '/resources/big/vectors_es_{}.bin'.format(size_es))
    for size_pt in VECTOR_SIZES:
        model_pt = word_vectors.load_model(PARENT_DIR + '/resources/big/vectors_pt_{}.bin'.format(size_pt))

        clf = classifier.build_classifier()

        T_path = PARENT_DIR + '/resources/big/trans_es_{}_pt_{}.npz'.format(size_es, size_pt)
        if os.path.exists(T_path):
            T = linear_trans.load_linear_transformation(T_path)
        else:
            lexicon = bilingual_lexicon.most_frequent_bilingual_lexicon_based_on_external_count(model_es.vocab,
                                                                                                model_pt.vocab)
            X, Y = zip(*word_vectors.bilingual_lexicon_vectors(model_es, model_pt, bilingual_lexicon=lexicon))
            T = linear_trans.linear_transformation(X, Y)
            linear_trans.save_linear_transformation(T_path, T)

        X, y = classifier.features_and_labels(friend_pairs, model_es, model_pt, T)

        measures = classifier.classify_with_cross_validation(X, y, clf=clf)

        print("{}\t{:0.4f}".format("es-{}-{}-1".format(size_es, size_pt), measures['Accuracy'][0]))

        clf = classifier.build_classifier()

        T_path = PARENT_DIR + '/resources/big/trans_pt_{}_es_{}.npz'.format(size_pt, size_es)
        if os.path.exists(T_path):
            T = linear_trans.load_linear_transformation(T_path)
        else:
            lexicon = bilingual_lexicon.most_frequent_bilingual_lexicon_based_on_external_count(model_es.vocab,
                                                                                                model_pt.vocab)
            X, Y = zip(*word_vectors.bilingual_lexicon_vectors(model_es, model_pt, bilingual_lexicon=lexicon))
            T = linear_trans.linear_transformation(X, Y, backwards=True)
            linear_trans.save_linear_transformation(T_path, T)

        X, y = classifier.features_and_labels(friend_pairs, model_es, model_pt, T, backwards=True)

        measures = classifier.classify_with_cross_validation(X, y, clf=clf)

        print("{}\t{:0.4f}".format("pt-{}-{}-1".format(size_pt, size_es), measures['Accuracy'][0]))
