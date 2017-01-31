from falsefriends import classifier
from falsefriends import word_vectors


def read_words(file_name):
    with open(file_name) as friends_file:
        friend_pairs = []
        for line in friends_file.readlines():
            word_es, word_pt, true_friends = line.split()
            if true_friends != '-1':
                true_friends = true_friends == '1'
                friend_pairs.append(classifier.FriendPair(word_es, word_pt, true_friends))
    return friend_pairs


def read_models(args_):
    model_es = word_vectors.load_model(args_.model_es_file_name)
    model_pt = word_vectors.load_model(args_.model_pt_file_name)
    return model_es, model_pt


def pairwise(iterate):
    iter_ = iter(iterate)
    return zip(iter_, iter_)
