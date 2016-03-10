# -*- coding: utf-8 -*-

import numpy as np
from scipy import spatial
from sklearn import cross_validation, svm, metrics


class FriendPair:
    def __init__(self, word_es, word_pt, true_friends):
        self.word_es = word_es
        self.word_pt = word_pt
        self.true_friends = true_friends


def classify_friends_and_predict(friend_pairs, model_es, model_pt, translation_matrix):
    friend_pairs = [friend_pair for friend_pair in friend_pairs
                    if friend_pair.word_es in model_es.vocab and friend_pair.word_pt in model_pt.vocab]

    vector_pairs = [[model_es[friend_pair.word_es], model_pt[friend_pair.word_pt]]
                    for friend_pair in friend_pairs]

    vectors_es, vectors_pt = zip(*vector_pairs)

    translations_es = np.dot(vectors_es, translation_matrix)

    distances = [spatial.distance.cosine(translation_es, vector_pt)
                 for (translation_es, vector_pt) in zip(translations_es, vectors_pt)]

    # noinspection PyPep8Naming
    X = [[distance] for distance in distances]
    y = [friend_pair.true_friends for friend_pair in friend_pairs]

    # noinspection PyPep8Naming
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)

    return metrics.precision_recall_fscore_support(y_test, y_predicted), metrics.accuracy_score(y_test, y_predicted)
