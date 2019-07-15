from imports import *
from helper_functions import *
from model import *

# read in data and glove embeddings
dat = pd.read_csv("../input/train.csv")
glove_file = "../input/glove.6B.50d.txt"
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_file)

# process data 
dat = data_process(dat, 'comment_text')

X = dat['comment_text'].values
Y = dat[['toxic', 'severe_toxic', 'obscene', 'threat',
         'insult', 'identity_hate']].values

X_padded = get_indices(X, word_to_index, 200)

X_train, X_test, y_train, y_test = train_test_split(X_padded, Y)


# compile model
model = basic_lstm((200,), word_to_vec_map, word_to_index, trainable=True)
model.summary()
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(X_train, y_train, epochs=1, batch_size=32, shuffle=True, validation_split=.1)


# get test set accuracy
get_test_accuracy(X_test, y_test)

