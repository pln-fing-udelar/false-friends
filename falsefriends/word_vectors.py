# -*- coding: utf-8 -*-


import word2vec


def train_model(infile, outfile):
    word2vec.word2vec(infile, outfile, size=100, verbose=True)
    model = word2vec.load(outfile)
    model.vocab




