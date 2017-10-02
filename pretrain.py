import argparse
from gensim.models.word2vec import Word2Vec, Text8Corpus
from gensim.models.word2vec import LineSentence

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('-d', '--dim', default=200, type=int)
    psr.add_argument('-p', '--path', default='ubuntu_data/train.txt')
    args = psr.parse_args()
    sentences = Text8Corpus(args.path)
    print('training')
    model = Word2Vec(sentences, size=args.dim, window=5, min_count=5, workers=4)
    model.save('ubuntu_word2vec_' + str(args.dim) + '.model')
    print('saved.')

if __name__ == '__main__': main()
