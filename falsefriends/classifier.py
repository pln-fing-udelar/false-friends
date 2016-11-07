# -*- coding: utf-8 -*-
import logging
from collections import defaultdict

import collections

import math
import numpy as np
from scipy import spatial
from sklearn import cross_validation, svm, metrics, preprocessing


class FriendPair:
    def __init__(self, word_es, word_pt, true_friends):
        self.word_es = word_es
        self.word_pt = word_pt
        self.true_friends = true_friends


def calculate_measures(tn, fp, fn, tp):
    return collections.OrderedDict([
        ('tn', tn),
        ('fp', fp),
        ('fn', fn),
        ('tp', tp),
        ('Neg. Precision', tn / (tn + fn) if tn + fn > 0 else 0),
        ('Neg. Recall', tn / (tn + fp) if tn + fp > 0 else 0),
        ('Neg. F1-score', tn / (tn + (fp + fn) / 2) if tn + fn + fp > 0 else 0),
        ('Precision', tp / (tp + fp) if tp + fp > 0 else 0),
        ('Recall', tp / (tp + fn) if tp + fn > 0 else 0),
        ('F1-score', tp / (tp + (fp + fn) / 2) if tp + fp + fn > 0 else 0),
        ('Accuracy', (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn > 0 else 0),
    ])


# noinspection PyPep8Naming
def classify_with_cross_validation(X, y, clf=svm.SVC(), n_folds=5):
    cv_matrices = []
    cv_measures = defaultdict(list)  # FIXME: use collections.OrderedDict too

    logging.info("classifying and predicting with cross validation")
    skf = cross_validation.StratifiedKFold(y, n_folds=n_folds)
    for train_indices, test_indices in skf:
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)

        confusion_matrix = metrics.confusion_matrix(y_test, y_predicted).flatten()
        cv_matrices.append(confusion_matrix)
        for measure_name, measure_value in calculate_measures(*confusion_matrix).items():
            cv_measures[measure_name].append(measure_value)

    for measure_name, measure_values in cv_measures.items():
        mean = np.mean(measure_values)
        delta = np.std(measure_values) * 1.96 / math.sqrt(n_folds)  # 95% of confidence
        cv_measures[measure_name] = (mean, delta)

    return cv_measures


# noinspection PyPep8Naming
def classify(X_train, X_test, y_train, y_test, clf=svm.SVC()):
    logging.info("classifying")
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)

    tn = sum(1 for test, predicted in zip(y_test, y_predicted) if not test and not predicted)
    fp = sum(1 for test, predicted in zip(y_test, y_predicted) if not test and predicted)
    fn = sum(1 for test, predicted in zip(y_test, y_predicted) if test and not predicted)
    tp = sum(1 for test, predicted in zip(y_test, y_predicted) if test and predicted)

    return calculate_measures(tn, fp, fn, tp)


# noinspection PyPep8Naming
def features_labels_and_scaler(friend_pairs, model_es, model_pt, translation_matrix, scaler=None, backwards=False):
    found_friend_pairs = [friend_pair for friend_pair in friend_pairs
                          if friend_pair.word_es in model_es.vocab and friend_pair.word_pt in model_pt.vocab]
    vector_pairs = [[model_es[friend_pair.word_es], model_pt[friend_pair.word_pt]]
                    for friend_pair in found_friend_pairs]
    vectors_es, vectors_pt = zip(*vector_pairs)
    if backwards:
        translations_pt = np.dot(vectors_pt, translation_matrix)
        distances = [spatial.distance.cosine(vector_es, translation_pt)
                     for (vector_es, translation_pt) in zip(vectors_es, translations_pt)]
    else:
        translations_es = np.dot(vectors_es, translation_matrix)
        distances = [spatial.distance.cosine(translation_es, vector_pt)
                     for (translation_es, vector_pt) in zip(translations_es, vectors_pt)]
    X = np.array([[distance] for distance in distances])
    y = np.array([friend_pair.true_friends for friend_pair in found_friend_pairs])
    if scaler is None:
        logging.info("scaling features")
        scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, y, scaler
