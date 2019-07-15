from imports import *
from preprocess import *

def basic_lstm(input_shape, word_to_vec_map, word_to_index, trainable=False):
    """
    creates the lstm graph using keras

    PARAMS
    ------------------------------------
    input_shape: shape of input (maxlen)
    word_to_vec_map: dictionary mapping words to their embeddings
    word_to_index: dictionary mapping words to their vocab indices
    """

    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index, trainable=trainable)
    embeddings = embedding_layer(sentence_indices)

    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embeddings)
    X = GobalMaxPool1D()(X)
    X = Dense(50, activation='relu')(X)
    X = Dropout(.1)(X)
    X = Dense(6, activation='sigmoid')(X)
    model = Model(inputs=sentence_indices, outputs=X)

    return model


