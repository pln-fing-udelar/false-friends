# -*- coding: utf-8 -*-

import word2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def train_model(in_file_name, out_file_name, use_plain_word2vec=False):
    if use_plain_word2vec:
        # noinspection PyCallingNonCallable
        word2vec.word2vec(in_file_name, out_file_name, verbose=True)
    else:
        model = Word2Vec(LineSentence(in_file_name), workers=4)
        model.save(out_file_name)
