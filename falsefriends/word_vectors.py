# -*- coding: utf-8 -*-

import word2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from falsefriends.bilingual_lexicon import bilingual_lexicon


def train_model(in_file_name, out_file_name, use_plain_word2vec=False):
    if use_plain_word2vec:
        # noinspection PyCallingNonCallable
        word2vec.word2vec(in_file_name, out_file_name, verbose=True)
    else:
        model = Word2Vec(LineSentence(in_file_name), workers=4)
        model.save(out_file_name)


def bilingual_lexicon_vectors(model_es_file_name, model_pt_file_name):
    model_es = Word2Vec.load(model_es_file_name)
    model_pt = Word2Vec.load(model_pt_file_name)
    return ((model_es[word_es], model_pt[word_pt])
            for word_es, word_pt in bilingual_lexicon()
            if word_es in model_es.vocab and word_pt in model_pt.vocab)
