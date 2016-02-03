#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import icu
from nltk.corpus import wordnet as wn
import unicodedata


def change_similar_letters(word):
    return unicodedata.normalize('NFKD', word).encode("ascii", "ignore").decode("ascii")


# words that not are "real" words (proper nouns, numbers, special characters)
def discartable(word):
    for letter in change_similar_letters(word):
        if not letter.isalpha() and letter not in ['_', '-']:
            return True
    if word[0].isupper():
        return True
    else:
        return False


def exact_matching():
    spa_words = set([word for word in wn.all_lemma_names(lang='spa') if not discartable(word)])
    por_words = set([word for word in wn.all_lemma_names(lang='por') if not discartable(word)])

    common_words = spa_words & por_words

    collator = icu.Collator.createInstance(icu.Locale('es_ES.UTF-8'))
    common_words_sorted = sorted(common_words, key=collator.getSortKey)
    return common_words_sorted
    # print('\n'.join(common_words_sorted))


# remove accents TODO replace ñ with nh
def similar_matching():
    # por_words = set(wn.all_lemma_names(lang='por'))

    spa_words = set([change_similar_letters(word) for word in wn.all_lemma_names(lang='spa') if not discartable(word)])
    por_words = set([change_similar_letters(word) for word in wn.all_lemma_names(lang='por') if not discartable(word)])

    common_words = spa_words & por_words

    collator = icu.Collator.createInstance(icu.Locale('es_ES.UTF-8'))
    common_words_sorted = sorted(common_words, key=collator.getSortKey)
    return common_words_sorted
    # print('\n'.join(common_words_sorted))

# baseline - wordnet's false friends
# ff = [x for x in set(wn.all_lemma_names(lang="spa")) & set(wn.all_lemma_names(lang="por")) if set(wn.synsets(x, lang="spa")) & set(wn.synsets(x, lang="por")) == set() ]

if __name__ == '__main__':
    f = open('equal_words.txt', 'w')
    f.write('\n'.join(exact_matching()))
    f.close()
    f = open('similar_words.txt', 'w')
    f.write('\n'.join(similar_matching()))
    f.close()
