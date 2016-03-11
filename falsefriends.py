#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import sys

from falsefriends import bilingual_lexicon, classifier, linear_trans, similar_words, wiki_parser, word_vectors

if __name__ == '__main__':
    def pairwise(iterate):
        _iter = iter(iterate)
        return zip(_iter, _iter)


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


    # noinspection PyUnusedLocal
    def command_bilingual_lexicon(_args):
        for lemma_name_spa, lemma_name_por in bilingual_lexicon.bilingual_lexicon():
            print("{} {}".format(lemma_name_spa, lemma_name_por))


    def command_lexicon_vectors(_args):
        for vector_es, vector_pt in word_vectors.bilingual_lexicon_vectors(_args.es_model_file_name,
                                                                           _args.pt_model_file_name):
            # TODO: save with all the precision
            print(' '.join(str(component) for component in vector_es))
            print(' '.join(str(component) for component in vector_pt))

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


    def read_words_and_models(_args):
        with open(_args.friends_file_name) as friends_file:
            friend_pairs = []
            for line in friends_file.readlines():
                word_es, word_pt, true_friends = line.split()
                true_friends = true_friends == '1'
                friend_pairs.append(classifier.FriendPair(word_es, word_pt, true_friends))
        model_es = word_vectors.load_model(_args.model_es_file_name)
        model_pt = word_vectors.load_model(_args.model_pt_file_name)
        return friend_pairs, model_es, model_pt


    def command_out_of_vocabulary(_args):
        friend_pairs, model_es, model_pt = read_words_and_models(_args)
        words_es = (friend_pair.word_es for friend_pair in friend_pairs)
        words_pt = (friend_pair.word_pt for friend_pair in friend_pairs)

        print("OOV es:")
        for word_es in word_vectors.words_out_of_vocabulary(model_es, words_es):
            print(word_es)

        print('')
        print("OOV pt:")
        for word_pt in word_vectors.words_out_of_vocabulary(model_pt, words_pt):
            print(word_pt)


    def command_classify(_args):
        friend_pairs, model_es, model_pt = read_words_and_models(_args)

        # noinspection PyPep8Naming
        T = linear_trans.load_linear_transformation(_args.translation_matrix_file_name)
        (precision, recall, f_score, support), accuracy = classifier.classify_friends_and_predict(friend_pairs,
                                                                                                  model_es,
                                                                                                  model_pt, T)
        print(precision, recall, f_score, support, accuracy)


    COMMANDS = {
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
        'bilingual_lexicon': {
            'function': command_bilingual_lexicon,
            'help': "print the Spanish-Portuguese bilingual lexicon",
            'parameters': [],
        },
        'lexicon_vectors': {
            'function': command_lexicon_vectors,
            'help': "print the vectors of the bilingual lexicon",
            'parameters': [
                {
                    'name': 'es_model_file_name',
                    'args': {},
                },
                {
                    'name': 'pt_model_file_name',
                    'args': {},
                },
            ]
        },
        'linear_trans': {
            'function': command_linear_trans,
            'help': "print the linear transformation for the input",
            'parameters': [],
        },
        'out_of_vocabulary': {
            'function': command_out_of_vocabulary,
            'help': "",
            'parameters': [
                {
                    'name': 'friends_file_name',
                    'args': {},
                },
                {
                    'name': 'model_es_file_name',
                    'args': {},
                },
                {
                    'name': 'model_pt_file_name',
                    'args': {},
                },
            ],
        },
        'classify': {
            'function': command_classify,
            'help': "classify word pairs of friends as false or true",
            'parameters': [
                {
                    'name': 'friends_file_name',
                    'args': {},
                },
                {
                    'name': 'model_es_file_name',
                    'args': {},
                },
                {
                    'name': 'model_pt_file_name',
                    'args': {},
                },
                {
                    'name': 'translation_matrix_file_name',
                    'args': {},
                },
            ],
        }
    }


    def args():
        _arg_parser = argparse.ArgumentParser()
        subparsers = _arg_parser.add_subparsers(dest='command', title='command')

        for command, command_values in sorted(COMMANDS.items()):
            sub_parser = subparsers.add_parser(command, help=command_values['help'])

            for parameter in command_values['parameters']:
                sub_parser.add_argument(parameter['name'], **parameter['args'])

        return _arg_parser, _arg_parser.parse_args()


    arg_parser, args = args()

    if args.command:
        # noinspection PyCallingNonCallable
        COMMANDS[args.command]['function'](args)
    else:
        arg_parser.print_help()
