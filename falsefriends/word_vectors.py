# -*- coding: utf-8 -*-
from gensim.models import Phrases, Word2Vec
from gensim.models.word2vec import LineSentence
import word2vec


def train_model(in_file_name, out_file_name, use_plain_word2vec=False, size=100, phrases_n_gram=1, threads=4):
    options = {
        'size': size,
    }

    if use_plain_word2vec:
        if phrases_n_gram > 1:
            phrases_file_name = '{}.phrases'.format(in_file_name)
            word2vec.word2phrase(in_file_name, phrases_file_name, verbose=True)
            in_file_name = phrases_file_name

        if threads:
            options['threads'] = threads

        # noinspection PyCallingNonCallable
        word2vec.word2vec(in_file_name, out_file_name, verbose=True, **options)
    else:
        sentences = LineSentence(in_file_name)
        for i in range(phrases_n_gram - 1):
            n_gram_transformer = Phrases(sentences)
            sentences = n_gram_transformer[sentences]

        if threads:
            options['workers'] = threads

        model = Word2Vec(sentences, **options)
        model.save(out_file_name)


def bilingual_lexicon_vectors(model_es, model_pt, bilingual_lexicon):
    return ((model_es[word_es], model_pt[word_pt])
            for word_es, word_pt in bilingual_lexicon
            if word_es in model_es.vocab and word_pt in model_pt.vocab)


def load_model(file_name):
    return Word2Vec.load(file_name)


def words_out_of_vocabulary(model, words):
    return (word for word in words if word not in model.vocab)
