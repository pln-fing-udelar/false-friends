# -*- coding: utf-8 -*-
import random

from nltk.corpus import wordnet as wn


def bilingual_lexicon():
    return ((lemma_name_spa, lemma_name_por)
            for synset in wn.all_synsets()
            for lemma_name_spa in synset.lemma_names(lang='spa')
            for lemma_name_por in synset.lemma_names(lang='por'))


# The default values don't work because WordNet doesn't have the lemma counts for Spanish and Portuguese.
def most_frequent_bilingual_lexicon(es_count_function=lambda lemma: lemma.count(),
                                    pt_count_function=lambda lemma: lemma.count()):
    # noinspection PyUnresolvedReferences
    return ((max((lemma for lemma in synset.lemmas(lang='spa')), key=es_count_function).name(),
             max((lemma for lemma in synset.lemmas(lang='por')), key=pt_count_function).name())
            for synset in wn.all_synsets()
            if synset.lemma_names(lang='spa') and synset.lemma_names(lang='por'))


def most_frequent_bilingual_lexicon_based_on_external_count(es_count_dict, pt_count_dict):
    return most_frequent_bilingual_lexicon(lambda lemma: es_count_dict[lemma.name()].count
                                           if lemma.name() in es_count_dict else 0,
                                           lambda lemma: pt_count_dict[lemma.name()].count
                                           if lemma.name() in pt_count_dict else 0)


def random_pair_per_synset_bilingual_lexicon():
    return ((random.choice(synset.lemma_names(lang='spa')), random.choice(synset.lemma_names(lang='por')))
            for synset in wn.all_synsets()
            if synset.lemma_names(lang='spa') and synset.lemma_names(lang='por'))
