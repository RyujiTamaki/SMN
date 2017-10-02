import argparse
import pandas as pd
import numpy as np
import joblib
from keras.layers import Dense, Input, Flatten, dot, concatenate, Reshape
from keras.layers import Conv2D, MaxPooling2D, Embedding, GRU
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, model_from_json

def build_SMN(max_turn, maxlen, word_dim, sent_dim, last_dim, num_words, embedding_matrix):
    context_input = Input(shape=(max_turn, maxlen), dtype='int32')
    response_input = Input(shape=(maxlen,), dtype='int32')

    embedding_layer = Embedding(num_words,
                            word_dim,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=True)
    sentence2vec = GRU(sent_dim, return_sequences=True)

    context_word_embedding = TimeDistributed(embedding_layer)(context_input)
    response_word_embedding = embedding_layer(response_input)

    context_sent_embedding = TimeDistributed(sentence2vec)(context_word_embedding)
    response_sent_embedding = sentence2vec(response_word_embedding)

    word_match = dot([context_word_embedding, response_word_embedding], axes=(3, 2))
    sent_match = dot([context_sent_embedding, response_sent_embedding], axes=(3, 2))

    word_match = Reshape((max_turn, 1, maxlen, maxlen))(word_match)
    sent_match = Reshape((max_turn, 1, maxlen, maxlen))(sent_match)
    match_2ch = concatenate([word_match, sent_match], axis=2)

    conv = TimeDistributed(Conv2D(8, (3, 3), activation='relu', data_format='channels_first'))(match_2ch)
    pool = TimeDistributed(MaxPooling2D(pool_size=(3, 3)))(conv)
    flat = TimeDistributed(Flatten())(pool)
    match = TimeDistributed(Dense(last_dim, activation='tanh'))(flat)

    out = GRU(last_dim)(match)
    output = Dense(1, activation='sigmoid')(out)

    model = Model(inputs=[context_input, response_input], outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('--maxlen', default=50, type=int)
    psr.add_argument('--max_turn', default=10, type=int)
    psr.add_argument('--num_words', default=50000, type=int)
    psr.add_argument('--word_dim', default=200, type=int)
    psr.add_argument('--sent_dim', default=200, type=int)
    psr.add_argument('--last_dim', default=100, type=int)
    psr.add_argument('--embedding_matrix', default='embedding_matrix.joblib')
    psr.add_argument('--train_data', default='train.joblib')
    psr.add_argument('--model_name', default='SMN_last')
    psr.add_argument('--batch_size', default=200, type=int)
    args = psr.parse_args()

    print('load data')
    embedding_matrix = joblib.load(args.embedding_matrix)
    train_data = joblib.load(args.train_data)

    print('build model')
    model = build_SMN(args.max_turn, args.maxlen, args.word_dim, args.sent_dim, args.last_dim, args.num_words, embedding_matrix)

    early_stopping =EarlyStopping(monitor='val_loss', patience=2)
    model_checkpoint = ModelCheckpoint(args.model_name + '.h5', save_best_only=True, save_weights_only=True)

    print('fitting')
    context = np.array(train_data['context'])
    response = np.array(train_data['response'])
    labels = train_data['labels']

    model.fit([context, response], labels,
    validation_split=0.05, batch_size=args.batch_size, epochs=200, callbacks=[early_stopping, model_checkpoint])
    json_string = model.to_json()
    open(args.model_name + '.json', 'w').write(json_string)

if __name__ == '__main__': main()
