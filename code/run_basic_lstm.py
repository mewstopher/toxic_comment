from imports import *
from preprocess import *
from model import *
dat = pd.read_csv("../input/train.csv")

dat['comment_text'] = dat['comment_text'].str.lower()

# remove punctuation
f = lambda x: ''.join([c for c in x if c not in punctuation])
dat['comment_text'] = dat.comment_text.apply(f)

# read in glove embeddings
glove_file = "../input/glove.6B.50d.txt"

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_file)

# get X_train and y_train from dataframe
# y is already a one-hot matrix for types of comments
# we are going to make X and Y numpy ndarrays
X = dat['comment_text'].values
Y = dat[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

X_train, X_test, y_train, y_test = train_test_split(X, Y)
maxLen = len(max(X))

# define model
model = basic_lstm((maxLen,), word_to_vec_map, word_to_index)

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)

model.fit(X_train_indices, y_train, epochs = 2, batch_size = 32, shuffle=True)

