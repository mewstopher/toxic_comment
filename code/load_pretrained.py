from imports import *
from helper_functions import *
from model import *
from bidirectional_lstm_mod import *
import sys

# read in data and glove embeddings using 300d
dat = pd.read_csv("../input/train.csv")

glove_file = "../input/glove.6B.300d.txt"
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_file)

# process data 
dat = data_process(dat, 'comment_text')

X = dat['comment_text'].values
Y = dat[['toxic', 'severe_toxic', 'obscene', 'threat',
         'insult', 'identity_hate']].values

X_padded = get_indices(X, word_to_index, 200)

X_train, X_test, y_train, y_test = train_test_split(X_padded, Y)

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])




# get test set accuracy
get_test_accuracy(X_test, y_test)

model = load_model("../output/bidirection_2epoch")

model.summary()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.summary()

x = Dense(50, activation='relu')(model.layers[-1].output)
o = Dense(6, activation='sigmoid')(x)
mod_input = Input(shape=(200,))
Model = Sequential()
model2 = Model(input=model.input, output=[o])
model2.summary()

model.fit(X_train, y_train, epochs=1, batch_size=32, shuffle=True, validation_split=.1)



