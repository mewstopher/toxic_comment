from imports import *
from preprocess import *
from model import *
dat = pd.read_csv("../input/train.csv")


# remove punctuation and tokenize 
dat['comment_text'] = dat['comment_text'].apply(lambda x: text_to_word_sequence(x))

# read in glove embeddings
glove_file = "../input/glove.6B.50d.txt"

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_file)

# get X_train and y_train from dataframe
# y is already a one-hot matrix for types of comments
# we are going to make X and Y numpy ndarrays
X = dat['comment_text'].values
Y = dat[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

X_train, X_test, y_train, y_test = train_test_split(X, Y)

# find max length of comments, padd with zeros, then truncate to 200
maxLen = len(max(X_train, key=len))
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)

X_train_padded = pad_sequences(X_train_indices, maxlen=200, truncating='post')

maxLen = 200
# define model
model = basic_lstm((maxLen,), word_to_vec_map, word_to_index)

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X_train_padded, y_train, epochs = 2, batch_size = 32, shuffle=True)


#training on entire training set takes too long for quick
#evaluations. shorten to first 100
X_tr2 = X_train_padded[0:100]
y_tr2 = y_train[0:100]

model.fit(X_tr2, y_tr2, epochs =1, batch_size = 50, shuffle=True)

X_test_indices = sentences_to_indices(X_test, word_to_index, 2000)
X_test_padded = pad_sequences(X_test_indices, maxlen=200, truncating='post')
pred = model.predict(X_test_padded[0:1])

C = 6
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
        x = X_test_indices
            num = np.argmax(pred[i])
                if(num != Y_test[i]):
                            print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())


