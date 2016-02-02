#!/usr/bin/env python3

import icu
from nltk.corpus import wordnet as wn

if __name__ == '__main__':
    spa_words = set(wn.all_lemma_names(lang='spa'))
    por_words = set(wn.all_lemma_names(lang='por'))

    common_words = spa_words & por_words

    collator = icu.Collator.createInstance(icu.Locale('es_ES.UTF-8'))
    common_words_sorted = sorted(common_words, key=collator.getSortKey)

    print('\n'.join(common_words_sorted))
