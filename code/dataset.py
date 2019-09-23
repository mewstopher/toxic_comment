from imports import *
from torch.utils.data import Dataset

class ToxicDataset(dataset):
    """
    class for toxic comment dataset
    """
    def __init__(self, toxic_csv_path, glove_path):
        self.df = pd.read_csv(toxic_csv_path)
        self.vocab_path = os.path.join(os.path.dirname(toxic_csv_path), "vocab.npy")
        self.vocab_vectors_path = os.path.join(os.path.dirname(toxic_csv_path), "vocab_vectors.npy")
        word_to_index, index_to_word, word_to_vec_map = self._read_glove_vecs(glove_path)
        self.emb_dim = np.int(self.word_to_vec_map['fox'].shape[0])

    def _build_vocab(self):
        if os.path.isfile(self.vocab_path) and os.path.isfile(self.vocab_vectors_path):
            print('using pre-built vocab')
            self.vocab = np.load(self.vocab_path, allow_pickle=True).item()
            self.initial_embeddings = np.load(self.vocab_vectors_path, allow_pickle=True).item
            self.unk_index = self.vocab['unk']
        print("creating vocab... this may take a few min")
        embeddings = []
        self.vocab = {}
        embeddings.append(np.zeros(self.emb_dim,))
        token_count = 0
        list_of_texts = self._tokenize_content('comment_text')
        words_in_sample = [text for comments in list_of_texts for text in comments]
        for word in words_in_sample:
            if word not in self.vocab:
                self.vocab[word] = token_count
                embeddings.append(._vec(word))
            else:
                if not unk_encountered:
                    embeddings.append(self._vec('unk'))
                    self.unk_index = token_count
                    self.unk_encountered = True
                    token_count +=1
                self.vocab[word] = self.unk_index
        if not unk_encountered:
            embeddings.append(self._vec('unk'))
            self.unk_index = token_count
            self.unk_encountered = True
            token_count +=1
            self.vocab[word] = self.unk_index
            self.initial_embeddings = np.array(embeddings)
            np.save(self.vocab_path, self.vocab)
            np.save(self.vocab_vectors_path, self.initial_embeddings)

    def _vec(self, w):
        return self.word_to_vec_map[w]



    def _read_glove_vecs(self, glove_path):
        """
        read in glove embeddings
        output 3 dictionaries:
        word_to_index, index_to_word, word_to_vec_map
        """
        with open(glove_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
                i = 1
                word_to_index = {}
                index_to_word = {}
                for w in sorted(words):
                    word_to_index[w] = i
                    index_to_word[i] = w
                    i += 1
        return word_to_index, index_to_word, word_to_vec_map

    def _tokenize_content(self, text_col):
        text_tokenized = self.df[text_col].astype(str).apply(lambda x: self.clean_special_char(x).split())
        return text_tokenized

    def clean_special_char(self, text):
        punc = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        for p in punc:
            text = text.replace(p, '')
        return text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_data = self.df.iloc[idx]
        text_tokenized = self._tokenize_content('comment_text')
        text_indices = [self.vocab.get(i, self.unk_index) for i in text_tokenized]
        text_len = len(text_indices)
        text_padded = F.pad(text_indices, (0, self.max_text_len - text_len), value=0, mode='constant')
        return text_indices_padded, labels


def _rread_glove_vecs(glove_path):
    """
    read in glove embeddings
    output 3 dictionaries:
    word_to_index, index_to_word, word_to_vec_map
    """
    with open(glove_path, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            i = 1
            word_to_index = {}
            index_to_word = {}
            for w in sorted(words):
                word_to_index[w] = i
                index_to_word[i] = w
                i += 1
    return word_to_index, index_to_word, word_to_vec_map

xx = _rread_glove_vecs("../input/glove_embeddings/Embeddings/glove.6B.50d.txt")

