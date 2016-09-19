#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import inspect
import logging

from sklearn import tree, svm, naive_bayes, neighbors

from falsefriends import bilingual_lexicon, classifier, linear_trans, similar_words, util, wiki_parser, word_vectors

if __name__ == '__main__':
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
        word_vectors.train_model(_args.input_file_name,
                                 _args.output_file_name,
                                 use_plain_word2vec=_args.use_plain_word2vec,
                                 phrases_n_gram=_args.phrases_n_gram,
                                 threads=_args.threads)


    # noinspection PyUnusedLocal
    def command_bilingual_lexicon(_args):
        for lemma_name_spa, lemma_name_por in bilingual_lexicon.bilingual_lexicon():
            print("{} {}".format(lemma_name_spa, lemma_name_por))

    # noinspection PyPep8Naming,PyUnusedLocal
    def command_linear_trans(_args):
        model_es = word_vectors.load_model(_args.model_es_file_name)
        model_pt = word_vectors.load_model(_args.model_pt_file_name)
        X, Y = zip(*word_vectors.bilingual_lexicon_vectors(model_es, model_pt))
        T = linear_trans.linear_transformation(X, Y)
        linear_trans.save_linear_transformation(_args.translation_matrix_file_name, T)


    def command_out_of_vocabulary(_args):
        friend_pairs = util.read_words(_args.friends_file_name)
        model_es, model_pt = util.read_models(_args)
        words_es = (friend_pair.word_es for friend_pair in friend_pairs)
        words_pt = (friend_pair.word_pt for friend_pair in friend_pairs)

        print("OOV es:")
        for word_es in word_vectors.words_out_of_vocabulary(model_es, words_es):
            print(word_es)

        print('')
        print("OOV pt:")
        for word_pt in word_vectors.words_out_of_vocabulary(model_pt, words_pt):
            print(word_pt)


    CLF_OPTIONS = {
        'DT': tree.DecisionTreeClassifier(),
        'GNB': naive_bayes.GaussianNB(),
        'kNN': neighbors.KNeighborsClassifier(),
        'SVM': svm.SVC(),
    }

    # noinspection PyShadowingNames
    def __print_metrics_matrix(measures):
        np = measures['Neg. Precision']
        nr = measures['Neg. Recall']
        nf = measures['Neg. F1-score']
        pp = measures['Precision']
        pr = measures['Recall']
        pf = measures['F1-score']

        print("               precision      recall         f1-score")
        print('')
        print("     False     {np:0.4f}         {nr:0.4f}         {nf:0.4f}".format(np=np, nr=nr, nf=nf))
        print("     True      {pp:0.4f}         {pr:0.4f}         {pf:0.4f}".format(pp=pp, pr=pr, pf=pf))
        print('')
        print("avg / total    {ap:0.4f}         {ar:0.4f}         {af:0.4f}".format(ap=(np + pp) / 2,
                                                                                    ar=(nr + pr) / 2,
                                                                                    af=(nf + pf) / 2))
        print('')


    def __print_confusion_matrix(measures):
        tn = measures['tn']
        fp = measures['fp']
        fn = measures['fn']
        tp = measures['tp']
        print("Confusion matrix")
        print('')
        print("\t\t(classified as)")
        print("\t\tTrue\tFalse")
        print("(are)\tTrue\t{tp:0.4f}\t{fn:0.4f}".format(tp=tp, fn=fn))
        print("(are)\tFalse\t{fp:0.4f}\t{tn:0.4f}".format(fp=fp, tn=tn))
        print('')


    # noinspection PyPep8Naming
    def command_classify(_args):
        training_friend_pairs = util.read_words(_args.training_friends_file_name)
        testing_friend_pairs = util.read_words(_args.testing_friends_file_name)
        model_es, model_pt = util.read_models(_args)

        T = linear_trans.load_linear_transformation(_args.translation_matrix_file_name)

        clf = CLF_OPTIONS[_args.classifier]

        if _args.cross_validation:
            X, y, _ = classifier.features_labels_and_scaler(training_friend_pairs + testing_friend_pairs, model_es,
                                                            model_pt, T)
            measures = classifier.classify_with_cross_validation(X, y, clf=clf)
            print('')

            print("Cross-validation measures with 95% of confidence:")

            for measure_name, (mean, delta) in measures.items():
                print("{measure_name}: {mean:0.4f} ± {delta:0.4f} --- [{inf:0.4f}, {sup:0.4f}]".format(
                    measure_name=measure_name, mean=mean, delta=delta, inf=mean - delta, sup=mean + delta))

            print('')

            mean_measures = {measure_name: mean for measure_name, (mean, delta) in measures.items()}
            __print_metrics_matrix(mean_measures)
            __print_confusion_matrix(mean_measures)
        else:
            X_train, y_train, scaler = classifier.features_labels_and_scaler(training_friend_pairs, model_es, model_pt,
                                                                             T)
            X_test, y_test, _ = classifier.features_labels_and_scaler(testing_friend_pairs, model_es, model_pt, T,
                                                                      scaler)
            measures = classifier.classify(X_train, X_test, y_train, y_test)

            print('')

            __print_metrics_matrix(measures)
            __print_confusion_matrix(measures)


    COMMANDS = collections.OrderedDict([
        (
            'similar_words',
            {
                'function': command_similar_words,
                'help': "write in files equal and similar words between Spanish and Portuguese",
                'parameters': [],
            }
        ),
        (
            'wiki_parser',
            {
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
            }
        ),
        (
            'word_vectors',
            {
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
                    {
                        'name': '--phrases-n-gram',
                        'args': {
                            'default': 1,
                            'type': int,
                        },
                    },
                    {
                        'name': '--threads',
                        'args': {
                            'default': inspect.signature(word_vectors.train_model).parameters['threads'].default,
                            'type': int,
                        },
                    }
                ],
            }
        ),
        (
            'bilingual_lexicon',
            {
                'function': command_bilingual_lexicon,
                'help': "print the Spanish-Portuguese bilingual lexicon",
                'parameters': [],
            }
        ),
        (
            'linear_trans',
            {
                'function': command_linear_trans,
                'help': "save the linear transformation given the models",
                'parameters': [
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
        ),
        (
            'out_of_vocabulary',
            {
                'function': command_out_of_vocabulary,
                'help': "print the words out of vocabulary of a friends list",
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
            }
        ),
        (
            'classify',
            {
                'function': command_classify,
                'help': "classify word pairs of friends as false or true",
                'parameters': [
                    {
                        'name': 'training_friends_file_name',
                        'args': {},
                    },
                    {
                        'name': 'testing_friends_file_name',
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
                    {
                        'name': '--classifier',
                        'args': {
                            'choices': sorted(list(CLF_OPTIONS.keys())),
                            'default': 'SVM',
                        },
                    },
                    {
                        'name': '--cross-validation',
                        'args': {
                            'action': 'store_const',
                            'const': True,
                            'default': False,
                        },
                    },
                ],
            }
        ),
    ])


    def args():
        _arg_parser = argparse.ArgumentParser()
        _arg_parser.add_argument(
            '-d', '--debug',
            help="Debug mode",
            action='store_const', dest='log_level', const=logging.DEBUG,
            default=logging.WARNING,
        )
        _arg_parser.add_argument(
            '-v', '--verbose',
            help="Verbose mde",
            action='store_const', dest='log_level', const=logging.INFO,
        )

        subparsers = _arg_parser.add_subparsers(dest='command', title='command')

        for command, command_values in COMMANDS.items():
            sub_parser = subparsers.add_parser(command, help=command_values['help'])

            for parameter in command_values['parameters']:
                sub_parser.add_argument(parameter['name'], **parameter['args'])

        return _arg_parser, _arg_parser.parse_args()


    arg_parser, args = args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=args.log_level)

    if args.command:
        # noinspection PyCallingNonCallable
        COMMANDS[args.command]['function'](args)
    else:
        arg_parser.print_help()
