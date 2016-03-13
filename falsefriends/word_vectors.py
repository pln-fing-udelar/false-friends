# -*- coding: utf-8 -*-

import word2vec
from gensim.models import Phrases, Word2Vec
from gensim.models.word2vec import LineSentence

from falsefriends.bilingual_lexicon import bilingual_lexicon


def train_model(in_file_name, out_file_name, use_plain_word2vec=False, phrases_n_gram=1):
    if use_plain_word2vec:
        if phrases_n_gram > 1:
            phrases_file_name = '{}.phrases'.format(in_file_name)
            word2vec.word2phrase(in_file_name, phrases_file_name, verbose=True)
            in_file_name = phrases_file_name
        # noinspection PyCallingNonCallable
        word2vec.word2vec(in_file_name, out_file_name, verbose=True)
    else:
        sentences = LineSentence(in_file_name)
        for i in range(phrases_n_gram - 1):
            n_gram_transformer = Phrases(sentences)
            sentences = n_gram_transformer[sentences]
        model = Word2Vec(sentences, workers=4)
        model.save(out_file_name)


def bilingual_lexicon_vectors(model_es_file_name, model_pt_file_name):
    model_es = load_model(model_es_file_name)
    model_pt = load_model(model_pt_file_name)
    return ((model_es[word_es], model_pt[word_pt])
            for word_es, word_pt in bilingual_lexicon()
            if word_es in model_es.vocab and word_pt in model_pt.vocab)


def load_model(file_name):
    return Word2Vec.load(file_name)


def words_out_of_vocabulary(model, words):
    return (word for word in words if word not in model.vocab)
