from imports import *
from torch.utils.data import Dataset

class ToxicDataset(Dataset):
    """
    class for toxic comment dataset
    """
    def __init__(self, toxic_csv_path, glove_path, vocab_path, embedding_path):
        self.df = pd.read_csv(toxic_csv_path)
        self.vocab_path = vocab_path
        self.vocab_vectors_path = embedding_path
        self.word_to_index, self.index_to_word, self.word_to_vec_map = self._read_glove_vecs(glove_path)
        self.emb_dim = np.int(self.word_to_vec_map['fox'].shape[0])
        self._build_vocab()
        self.max_text_len = len(max(self.df.comment_text))

    def _build_vocab(self):
        if os.path.isfile(self.vocab_path) and os.path.isfile(self.vocab_vectors_path):
            print('using pre-built vocab')
            self.vocab = np.load(self.vocab_path, allow_pickle=True).item()
            self.initial_embeddings = np.load(self.vocab_vectors_path, allow_pickle=True)
            self.unk_index = self.vocab['unk']
        else:
            print("creating vocab... this may take a few min")
            embeddings = []
            self.vocab = {}
            self.unkown_words = []
            unk_encountered = False
            embeddings.append(np.zeros(self.emb_dim,))
            token_count = 0
            list_of_texts = self._tokenize_content('comment_text')
            words_in_sample = [text for comments in list_of_texts for text in comments]
            for word in words_in_sample:
                if word not in self.vocab:
                    self.vocab[word] = token_count
                    if self._vec(word) != 'unkown word':
                        embeddings.append(self._vec(word))
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
        try:
            word_as_vec = self.word_to_vec_map[w]
        except:
           self.unkown_words.append(w)
           word_as_vec = "unkown word"
        return word_as_vec




    def _read_glove_vecs(self, glove_path):
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

    def _tokenize_content(self, text_col):
        text_tokenized = self.df[text_col].astype(str).apply(lambda x: self.clean_special_char(x).lower().split())
        return text_tokenized

    def _tokenize_sample(self, content):
        sample_tokenized = self.clean_special_char(content).lower().split()
        return sample_tokenized

    def clean_special_char(self, text):
        punc = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        for p in punc:
            text = text.replace(p, '')
        return text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_data = self.df.iloc[idx]
        text_tokenized = self._tokenize_sample(sample_data['comment_text'])
        text_indices = torch.tensor([self.vocab.get(i, self.unk_index) for i in text_tokenized],dtype=torch.long)
        text_len = len(text_indices)
        labels = torch.tensor(sample_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.astype(float), dtype=torch.float32)
        text_padded = F.pad(text_indices, (0, self.max_text_len - text_len), value=0, mode='constant')
        return text_padded, labels


