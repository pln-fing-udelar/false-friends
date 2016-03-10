from gensim.models import Word2Vec
from bilingual_lexicon import bilingual_lexicon

def generate_linear_trans_input(vectors_orig, vectors_dest, dest_file):
    f.open(dest_file, 'w')
    model_orig = Word2Vec.load(vectors_es, binary=True)
    model_dest = Word2Vec.load(vectors_pt, binary=True)
    for word_pair in bilingual_lexicon():
        f.write(' '.join(str(x) for x in model_orig[word_pair[0]]))
        f.write(' '.join(str(x) for x in model_dest[word_pair[1]]))
    f.close()
