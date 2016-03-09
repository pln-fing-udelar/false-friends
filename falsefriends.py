#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import sys

from falsefriends import bilingual_lexicon, linear_trans, similar_words, wiki_parser, word_vectors

if __name__ == '__main__':
    def pairwise(iterate):
        _iter = iter(iterate)
        return zip(_iter, _iter)


    # noinspection PyUnusedLocal
    def command_bilingual_lexicon(_args):
        for lemma_name_spa, lemma_name_por in bilingual_lexicon.bilingual_lexicon():
            print("{} {}".format(lemma_name_spa, lemma_name_por))

    # noinspection PyPep8Naming,PyUnusedLocal
    def command_linear_trans(_args):
        lines = sys.stdin.readlines()

        X = []
        Y = []
        for line1, line2 in pairwise(lines):
            X.append([float(coord) for coord in line1.split()])
            Y.append([float(coord) for coord in line2.split()])

        transformation = linear_trans.linear_transformation(X, Y)
        np.savetxt(sys.stdout.buffer, transformation)


    # noinspection PyUnusedLocal
    def command_similar_words(_args):
        with open('resources/equal_words.txt', 'w') as file:
            for line in similar_words.exact_matching():
                file.write(line + '\n')
        with open('resources/similar_words.txt', 'w') as file:
            for line in similar_words.similar_matching():
                file.write(line + '\n')


    def command_wiki_parser(_args):
        wiki_parser.pre_process_wiki(_args.input_file_name, _args.output_file_name, _args.lang)


    def command_word_vectors(_args):
        word_vectors.train_model(_args.input_file_name, _args.output_file_name,
                                 use_plain_word2vec=_args.use_plain_word2vec)


    COMMANDS = {
        'bilingual_lexicon': {
            'function': command_bilingual_lexicon,
            'help': "print the Spanish-Portuguese bilingual lexicon",
            'parameters': [],
        },
        'linear_trans': {
            'function': command_linear_trans,
            'help': "print the linear transformation for the input",
            'parameters': [],
        },
        'similar_words': {
            'function': command_similar_words,
            'help': "write in files equal and similar words between Spanish and Portuguese",
            'parameters': [],
        },
        'wiki_parser': {
            'function': command_wiki_parser,
            'help': "output the pre-processed Wikipedia passed as input",
            'parameters': [
                {
                    'name': 'input_file_name',
                    'args': {},
                },
                {
                    'name': 'output_file_name',
                    'args': {},
                },
                {
                    'name': 'lang',
                    'args': {},
                },
            ],
        },
        'word_vectors': {
            'function': command_word_vectors,
            'help': "calculate the vector space from sentences",
            'parameters': [
                {
                    'name': 'input_file_name',
                    'args': {},
                },
                {
                    'name': 'output_file_name',
                    'args': {},
                },
                {
                    'name': '--use-plain-word2vec',
                    'args': {
                        'action': 'store_const',
                        'const': True,
                        'default': False,
                    },
                },
            ],
        },
    }


    def args():
        arg_parser = argparse.ArgumentParser()
        subparsers = arg_parser.add_subparsers(dest='command', title='command')

        for command, command_values in sorted(COMMANDS.items()):
            sub_parser = subparsers.add_parser(command, help=command_values['help'])

            for parameter in command_values['parameters']:
                sub_parser.add_argument(parameter['name'], **parameter['args'])

        return arg_parser.parse_args()


    args = args()

    # noinspection PyCallingNonCallable
    COMMANDS[args.command]['function'](args)
