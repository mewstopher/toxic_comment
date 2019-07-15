from imports import *
from helper_functions import *

def data_process(dataframe, text_col):
    """
    removes punctuation, tokenizes from a
    pandas dataframe

    PARAMS
    ---------------------------
    dat: pd dataframe
    """
    dataframe[text_col] = dataframe[text_col].apply(lambda x: text_to_word_sequence(x))
    return dataframe

def get_indices(X, word_to_index, maxlen):
    """
    get indices from comments, then padd comments

    PARAMS
    ------------------------
    X: values from dataset (comments)
    word_to_index: dictionary mapping words to
    indices
    maxlen: length to truncate comments at
    """
    max_sentence_length = len(max(X, key=len))
    indices = sentences_to_indices(X, word_to_index,
                                   max_sentence_length)
    X_padded = pad_sequences(indices, maxlen=maxlen,
                             truncating='post')
    return X_padded





