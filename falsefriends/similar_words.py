# -*- coding: utf-8 -*-

import icu
from nltk.corpus import wordnet as wn
import unicodedata


def change_similar_letters(word):
    return unicodedata.normalize('NFKD', word).encode("ascii", "ignore").decode("ascii")


def discartable(word):
    """Discards words that are not "real" words (proper nouns, numbers, special characters, etc.).
    :param word:
    """
    for letter in change_similar_letters(word):
        if not letter.isalpha() and letter not in ['_', '-']:
            return True
    if word[0].isupper():
        return True
    else:
        return False


def exact_matching():
    spa_words = {word for word in wn.all_lemma_names(lang='spa') if not discartable(word)}
    por_words = {word for word in wn.all_lemma_names(lang='por') if not discartable(word)}

    common_words = spa_words & por_words

    collator = icu.Collator.createInstance(icu.Locale('es_ES.UTF-8'))
    return sorted(common_words, key=collator.getSortKey)


# TODO: replace ñ with nh, and etc.
def similar_matching():
    """Matches ignoring accent."""
    spa_words = {change_similar_letters(word) for word in wn.all_lemma_names(lang='spa') if not discartable(word)}
    por_words = {change_similar_letters(word) for word in wn.all_lemma_names(lang='por') if not discartable(word)}

    common_words = spa_words & por_words

    collator = icu.Collator.createInstance(icu.Locale('es_ES.UTF-8'))
    return sorted(common_words, key=collator.getSortKey)


def baseline():
    spa_words = {word for word in wn.all_lemma_names(lang='spa') if not discartable(word)}
    por_words = {word for word in wn.all_lemma_names(lang='por') if not discartable(word)}

    return [x for x in spa_words & por_words if
            set(wn.synsets(x, lang="spa")) & set(wn.synsets(x, lang="por")) == set()]
