import numpy as np
import joblib
import argparse
import codecs
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def build_multiturn_data(multiturn_data):
    contexts = []
    responses = []
    labels = []
    with codecs.open(multiturn_data,'r','utf-8') as f:
        for line in tqdm(f):
            line = line.replace('_','')
            parts = line.strip().split('\t')

            lable = parts[0]
            message = ''
            for i in range(1, len(parts)-1, 1):
                message += parts[i]
                message += ' _eot_ '

            response = parts[-1]

            contexts.append(message)
            responses.append(response)
            labels.append(lable)

    return contexts, responses, np.array(labels)

def preprocess_texts(texts, maxlen):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen)

def preprocess_multi_turn_texts(context, max_turn, maxlen):
    multi_turn_texts = []
    for i in tqdm(range(len(context))):
        multi_turn_texts.append(context[i].split('_eot_')[-(max_turn+1):-1])
        if len(multi_turn_texts[i]) <= max_turn:
            tmp = multi_turn_texts[i][:]
            multi_turn_texts[i] = [' '] * (max_turn - len(multi_turn_texts[i]))
            multi_turn_texts[i].extend(tmp)

    multi_turn_texts = [preprocess_texts(multi_turn_texts[i], maxlen) for i in tqdm(range(len(multi_turn_texts)))]
    return multi_turn_texts

def word2vec_embedding(path, num_words, embedding_dim, word_index):
    w2v = Word2Vec.load(path)
    num_words = min(num_words, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, embedding_dim))
    for word, i in word_index.items():
        if i > num_words:
            continue
        try:
            embedding_matrix[i] = w2v[word]
        except KeyError:
            pass
    return embedding_matrix

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('--maxlen', default=50, type=int)
    psr.add_argument('--max_turn', default=10, type=int)
    psr.add_argument('--num_words', default=50000, type=int)
    psr.add_argument('--embedding_dim', default=200, type=int)
    psr.add_argument('--w2v_path', default='ubuntu_word2vec_200.model')
    psr.add_argument('--train_data', default='ubuntu_data/train.txt')
    psr.add_argument('--test_data', default='ubuntu_data/test.txt')
    args = psr.parse_args()

    print('load data')
    train_context, train_response, train_labels = build_multiturn_data(args.train_data)
    test_context, test_response, test_labels = build_multiturn_data(args.test_data)

    print('tokenize')
    global tokenizer, maxlen
    tokenizer = Tokenizer(num_words=args.num_words)
    tokenizer.fit_on_texts(np.append(train_context, train_response))
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    print('create word matrix')
    embedding_matrix = word2vec_embedding(path=args.w2v_path, num_words=args.num_words, embedding_dim=args.embedding_dim, word_index=word_index)

    print('preprocess data')
    train_context = preprocess_multi_turn_texts(train_context, args.max_turn, args.maxlen)
    train_response = preprocess_texts(train_response, args.maxlen)
    test_context = preprocess_multi_turn_texts(test_context, args.max_turn, args.maxlen)
    test_response = preprocess_texts(test_response, args.maxlen)

    train_data = {'context': train_context, 'response': train_response, 'labels': train_labels}
    test_data = {'context': test_context, 'response': test_response, 'labels': test_labels}

    print('dump')
    joblib.dump(train_data, 'train.joblib', protocol=-1, compress=3)
    joblib.dump(test_data, 'test.joblib', protocol=-1, compress=3)
    joblib.dump(embedding_matrix, 'embedding_matrix.joblib', protocol=-1, compress=3)

if __name__ == '__main__': main()
