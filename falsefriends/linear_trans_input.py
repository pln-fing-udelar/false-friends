from gensim.models import Word2Vec
from falsefriends.bilingual_lexicon import bilingual_lexicon

def generate_linear_trans_input(vectors_orig, vectors_dest, dest_file):
    f = open(dest_file, 'w')
    model_orig = Word2Vec.load(vectors_orig)
    model_dest = Word2Vec.load(vectors_dest)
    for word_pair in bilingual_lexicon():
        if word_pair[0] in model_orig.vocab and word_pair[1] in model_dest.vocab:
            f.write(' '.join(str(x) for x in model_orig[word_pair[0]]))
            f.write(' '.join(str(x) for x in model_dest[word_pair[1]]))
    f.close()
