# library
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from tqdm import tqdm
import re
import fasttext
import warnings
import math
warnings.filterwarnings('ignore')

# modeling library
import tensorflow as tf
from keras.models import Model, Sequential,load_model
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Input, Dense, LSTM, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split

# data path
big_model_path = './model/FINAL_MODELS/final_big_0520_wiki_emb.h5'
test_path = './data/final_data/0518_ked_test_only.pkl'

# embedding
fasttext_model_cache = {}
fasttext_model_path = './ked_fasttext_train_test_300.bin'   # fast text embedding
if fasttext_model_path not in fasttext_model_cache:
  fasttext_model_cache[fasttext_model_path] = fasttext.load_model(fasttext_model_path)
fasttext_model = fasttext_model_cache[fasttext_model_path]

EMBEDDING_DIM = 300  # fasttext embedding dimension

def drop_token(x):
    sw = ['부대사업', '일체', '사업', '부대', '각호', '판매업', '한다', '거나', '에게', '관련', '목적']  # stopwords
    # , '여', '왠', '토', '공', '엉', '도', '업', '투', '위', '내', '호에',
    x = [word for word in x if not word in sw]
    if len(x) > 50:
        return x[:50]
    else:
        return x

def read_test_corpus(path):
    data = pd.read_pickle(path)
    ked_test_X = [x for x in data.iloc[:, -1].apply(drop_token)]
    ked_test_y = pd.get_dummies(data['big']).values
    return np.array(ked_test_X), np.array(ked_test_y)


def padding(tokenized_sentence, max_len=50):
    if len(tokenized_sentence) >= max_len:
        padded_sentence = tokenized_sentence[:max_len]
    else:
        n_to_pad = max_len - len(tokenized_sentence)
        padded_sentence = tokenized_sentence + [''] * n_to_pad
    return padded_sentence

def get_word_vectors(words):
    result = []
    for word in words:
        if not word:
            result.append(np.zeros((EMBEDDING_DIM,)))
        else:
            result.append(fasttext_model.get_word_vector(word))
    return np.array(result)

def make_input(padded_sentence):
    word_vectors = [get_word_vectors(padded_sentence)]
    return np.array(word_vectors)



def pad(data, max_len=50):
    if max_len == 0:
        max_len = max(len(tokens) for tokens in data)
    result = []
    for tokens in tqdm(data, desc='Padding'):
        if len(tokens) >= max_len:
            result.append(tokens[:max_len])
        else:
            n_to_pad = max_len - len(tokens)
            result.append(tokens + [''] * n_to_pad)
    return max_len, result


def preprocess(tokenized_sentences):
    max_tokens, padded_sentences = pad(tokenized_sentences)
    return padded_sentences


## model class
class Dataset(tf.keras.utils.Sequence):
    fasttext_model_cache = {}

    def __init__(self, x_set, batch_size, fasttext_model):
        self.x_set = x_set
        # self.y_set = y_set
        self.batch_size = batch_size
        self.fasttext_model = fasttext_model

    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)  # ceil : 올림

    def __getitem__(self, idx):
        padded_sentences = self.x_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        word_vectors = [self.get_word_vectors(padded_sentence) for padded_sentence in padded_sentences]
        # batch_y = self.y_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(word_vectors)  # , np.array(batch_y)

    ## word_vectors를 얻기
    def get_word_vectors(self, words):
        result = []
        for word in words:
            if not word:
                result.append(np.zeros((EMBEDDING_DIM,)))  # LSTM을 위한 zero padding
            else:
                result.append(self.fasttext_model.get_word_vector(word))
        return np.array(result)