from imports import *
from helper_functions import *
from model import *
from bidirectional_lstm_mod import *
import sys

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



# for running as script
if __name__ == "__main__":
    load_or_save = input("press l to load model, press t to train from scratch: ")
    train_obs = input("number of observations to use for testing(type 'a' for all): ")
        if train_obs == 'a':
            train_obs = X_train.shape[0]
        else:
            train_obs = int(train_obs)
    if load_or_save == 't':
        model = trial_lstm((200,), word_to_vec_map, word_to_index, trainable=True)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train[0:train_obs], y_train[0:train_obs], epochs=1, batch_size=32, shuffle=True)
    else:
        print(os.listdir("../output/"))
        model_to_load = input("type in model name to load: ")
        model = load_model("../output/" + model_to_load)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        model.fit(X_train[0:train_obs], y_train[0:train_obs], epochs=1, batch_size=32, shuffle=True)
    do_save = input("save model? press y to save: ")
    if do_save == 'y':
        model_name = input("type in name of new model to save: ")
        model.save("../output/" + model_name)
    do_test = input("test model? press y to test: ")
    if do_test == 'y':
        get_test_accuracy(X_test, y_test, model=model)

# submit predictions to csv
    to_save = input('save sumbittion: ')
    if to_save == 'y':
        test_dat = pd.read_csv("../input/test.csv")
        test_dat = data_process(test_dat, 'comment_text')

        X_te = test_dat['comment_text'].values
        X_test_padded = get_indices(X_te, word_to_index, 200)
        y_test = model.predict([X_test_padded], batch_size=1024, verbose=1)
        sample_submission = pd.read_csv('../input/sample_submission.csv')
        sample_submission[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = y_test
        sample_submission.to_csv('../output/submission_X.csv', index=False)

    sys.exit()



# fit model
# compile model
model = trial_lstm((200,), word_to_vec_map, word_to_index, trainable=True)
model.summary()
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])


model.fit(X_train[0:80], y_train[0:80], epochs=1, batch_size=32, shuffle=True, validation_split=.1)


# get test set accuracy
get_test_accuracy(X_test, y_test)

