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

    X = LSTM(128,  return_sequences=True)(embeddings)
    X = Dropout(.5)(X)
    X = LSTM(128)(X)
    X = Dropout(.5)(X)
    X = Dense(6, activation='sigmoid')(X)
    model = Model(inputs=sentence_indices, outputs=X)

    return model


