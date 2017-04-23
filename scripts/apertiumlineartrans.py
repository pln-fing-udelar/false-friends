#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import subprocess
import sys

import numpy as np

PARENT_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..')
sys.path.insert(0, PARENT_DIR)

from falsefriends import classifier, linear_trans, util, word_vectors

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_es = word_vectors.load_model(PARENT_DIR + '/resources/big/vectors_es_100.bin')
model_pt = word_vectors.load_model(PARENT_DIR + '/resources/big/vectors_pt_100.bin')

n_top = 50000
spanish_top = model_es.index2word[:n_top]

with subprocess.Popen(['apertium', 'es-pt'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True) \
        as process:
    translated_spanish_top_str, _ = process.communicate('\n'.join(spanish_top))

# noinspection PyTypeChecker
translated_spanish_top = translated_spanish_top_str.split('\n')

# noinspection PyTypeChecker
lexicon = [pair for pair in zip(spanish_top, translated_spanish_top) if not pair[1].startswith('*')]

print("{} of the top {} Spanish words could be translated with Apertium".format(len(lexicon), n_top))

X, Y = zip(*word_vectors.bilingual_lexicon_vectors(model_es, model_pt, lexicon))

# noinspection PyRedeclaration
print("{}/{} have associated vectors".format(sum(1 for _ in X), len(lexicon)))

X_array = np.vstack(X)
Y_array = np.vstack(Y)

logging.info("Computing linear transformations and classifying with cross-validation...")
print('')
print("   N\t Acc")

friend_pairs = util.read_words(PARENT_DIR + '/resources/sepulveda2011_original.txt')

clf = classifier.build_classifier()

for size in range(50, len(X), 50):
    indices = np.random.choice(list(range(X_array.shape[0])), size, replace=False)

    T = linear_trans.linear_transformation(list(X_array[indices, :]), list(Y_array[indices, :]))

    X, y = classifier.features_and_labels(friend_pairs, model_es, model_pt, T)

    measures = classifier.classify_with_cross_validation(X, y, clf=clf)

    print("{: >5}\t{:0.4f}".format(size, measures['Accuracy'][0]))
