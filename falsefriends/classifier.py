# -*- coding: utf-8 -*-
import collections
import logging
import math

from nltk.corpus import wordnet as wn
import numpy as np
from scipy import spatial
from sklearn import cross_validation, metrics, pipeline, preprocessing, svm


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


def build_classifier(base_clf=svm.SVC()):
    # The imputer is for "use_taxonomy", and shouldn't affect if it's False.
    # TODO: should also try with other imputer strategies
    return pipeline.make_pipeline(preprocessing.Imputer(strategy='most_frequent'), preprocessing.StandardScaler(),
                                  base_clf)


# noinspection PyPep8Naming
def classify_with_cross_validation(X, y, clf, n_folds=5):
    cv_matrices = []
    cv_measures = collections.defaultdict(list)  # FIXME: use collections.OrderedDict too

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
def classify(X_train, X_test, y_train, y_test, clf):
    logging.info("classifying")
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)

    tn = sum(1 for test, predicted in zip(y_test, y_predicted) if not test and not predicted)
    fp = sum(1 for test, predicted in zip(y_test, y_predicted) if not test and predicted)
    fn = sum(1 for test, predicted in zip(y_test, y_predicted) if test and not predicted)
    tp = sum(1 for test, predicted in zip(y_test, y_predicted) if test and predicted)

    return calculate_measures(tn, fp, fn, tp)


def top(x, friend_pairs, model_es, model_pt):
    return sorted(friend_pairs,
                  key=lambda f: min(model_es.vocab[f.word_es].count if f.word_es in model_es.vocab else 0,
                                    model_pt.vocab[f.word_pt].count if f.word_pt in model_pt.vocab else 0),
                  reverse=True)[:round(len(friend_pairs) * x)]


# noinspection PyPep8Naming
def features_and_labels(friend_pairs, model_es, model_pt, translation_matrix, backwards=False, topx=None,
                        use_taxonomy=False):
    logging.info("computing features")

    models = {
        'es': model_es,
        'pt': model_pt,
    }

    if topx:
        found_friend_pairs = top(topx, friend_pairs, model_es, model_pt)
    else:
        found_friend_pairs = [friend_pair for friend_pair in friend_pairs
                              if friend_pair.word_es in model_es.vocab and friend_pair.word_pt in model_pt.vocab]

        logging.info('{} pairs have words that were not found in the corpus'.format(
            len(friend_pairs) - len(found_friend_pairs)))

    words = {
        'es': [friend_pair.word_es for friend_pair in found_friend_pairs],
        'pt': [friend_pair.word_pt for friend_pair in found_friend_pairs],
    }
    vectors = {lang: models[lang][word_lang] for lang, word_lang in words.items()}

    if backwards:
        source = 'pt'
        target = 'es'
    else:
        source = 'es'
        target = 'pt'

    translations = np.dot(vectors[source], translation_matrix)
    distances = [spatial.distance.cosine(translation, vector_target)
                 for (translation, vector_target) in zip(translations, vectors[target])]

    ordinals = (len(np.where(models[target].similar_by_word(vector_target, topn=None) > distance)[0]) - 1
                for vector_target, distance in zip(vectors[target], distances))

    distances_closest = (sum(spatial.distance.cosine(np.dot(models[source][similar_word],
                                                            translation_matrix), vector_target)
                             for similar_word, _ in models[source].similar_by_word(word_source, topn=5))
                         for word_source, vector_target in zip(words[source], vectors[target]))

    # Gives better results (from 76.98% to 79.42%).
    # frequencies = (min((models[source].vocab[word_source].index / len(models[source].vocab))
    #                    / (models[target].vocab[word_target].index / len(models[target].vocab)),
    #                    (models[target].vocab[word_target].index / len(models[target].vocab)
    #                    / (models[source].vocab[word_source].index / len(models[source].vocab))))
    #                for word_source, word_target in zip(words[source], words[target]))

    # SHARE_WINDOW = 5
    # closest_shared_count = (sum((Counter(model_pt.similar_by_vector(np.dot(model_es[similar_word],
    #                                                                        translation_matrix), topn=1)[0][0]
    #                                     for similar_word, _ in model_es.similar_by_word(word_source,
    #                                                                                     topn=SHARE_WINDOW))
    #                             & Counter(similar_word for similar_word, _
    #                                       in model_pt.similar_by_vector(vector_target, topn=SHARE_WINDOW + 1))
    #                             ).values())
    #                        for word_source, vector_target in zip(words[source], vectors[target]))

    if use_taxonomy:
        logging.info("{} pairs have words that are not in WordNet".format(
            sum(1 for friend_pair in found_friend_pairs
                if not wn.synsets(friend_pair.word_es, lang='spa')
                and not wn.synsets(friend_pair.word_pt, lang='por'))))

        lch_similarities = [max(((wn.lch_similarity(synset_es, synset_pt) or 0) if synset_es.pos == synset_pt.pos else 0
                                 for synset_es in wn.synsets(friend_pair.word_es, lang='spa')
                                 for synset_pt in wn.synsets(friend_pair.word_pt, lang='por')), default=np.nan)
                            for friend_pair in found_friend_pairs]
        wup_similarities = [max(((wn.wup_similarity(synset_es, synset_pt) or 0) if synset_es.pos == synset_pt.pos else 0
                                 for synset_es in wn.synsets(friend_pair.word_es, lang='spa')
                                 for synset_pt in wn.synsets(friend_pair.word_pt, lang='por')), default=np.nan)
                            for friend_pair in found_friend_pairs]

        X = np.array(list(zip(distances, ordinals, distances_closest, lch_similarities, wup_similarities)))
    else:
        X = np.array(list(zip(distances, ordinals, distances_closest)))

    y = np.array([friend_pair.true_friends for friend_pair in found_friend_pairs])
    return X, y
