# -*- coding: utf-8 -*-

from nltk.corpus import wordnet as wn


def bilingual_lexicon():
    return ((lemma_name_spa, lemma_name_por)
            for synset in wn.all_synsets()
            for lemma_name_spa in synset.lemma_names(lang='spa')
            for lemma_name_por in synset.lemma_names(lang='por'))
