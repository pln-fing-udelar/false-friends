#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import icu
from nltk.corpus import wordnet as wn
import unicodedata

def exact_matching():
    spa_words = set(wn.all_lemma_names(lang='spa'))
    por_words = set(wn.all_lemma_names(lang='por'))

    common_words = spa_words & por_words

    collator = icu.Collator.createInstance(icu.Locale('es_ES.UTF-8'))
    common_words_sorted = sorted(common_words, key=collator.getSortKey)

    return common_words
    # print('\n'.join(common_words_sorted))


#
def similar_matching():
    # por_words = set(wn.all_lemma_names(lang='por'))
    def change_similar_letters(word):
        return unicodedata.normalize('NFKD', word).encode("ascii", "ignore").decode("ascii")
    
    spa_words = set([change_similar_letters(word) for word in wn.all_lemma_names(lang='spa')])
    por_words = set([change_similar_letters(word) for word in wn.all_lemma_names(lang='por')])

    common_words = spa_words & por_words

    collator = icu.Collator.createInstance(icu.Locale('es_ES.UTF-8'))
    common_words_sorted = sorted(common_words, key=collator.getSortKey)
    return common_words
    # print('\n'.join(common_words_sorted))

# baseline - check if
# ff = [x for x in set(wn.all_lemma_names(lang="spa")) & set(wn.all_lemma_names(lang="por")) if set(wn.synsets(x, lang="spa")) & set(wn.synsets(x, lang="por")) == set() ]
