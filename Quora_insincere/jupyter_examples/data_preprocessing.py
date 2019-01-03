import os
import pandas as pd
import numpy as np
import re
import json

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

## some config values 
embed_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use

DATA_DIR = '../input/'
TRAIN_DATA = 'train.npy'
TRAIN_LEN_DATA = 'train_len.npy'
TEST_DATA = 'test.npy'
TEST_LEN_DATA = 'test_len.npy'
TEST_ID_DATA = 'test_id.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
DATA_CONFIGS = 'data_configs.json'

# puncts = [ '"', ')', '(', '-', '|', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
#  '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
#  '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
#  '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
#  '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', 
# 'é', '&amp;','₹', 'á', '²', 'ế', '청', '하', '¨', '‘', '√', '\xa0', '高', '端', '大', '气', '上', '档', '次', '_', '½', 'π', '#', 
# '小', '鹿', '乱', '撞', '成', '语', 'ë', 'à', 'ç', '@', 'ü', 'č', 'ć', 'ž', 'đ', '°', 'द', 'े', 'श', '्', 'र', 'ो', 'ह', 
# 'ि', 'प', 'स', 'थ', 'त', 'न', 'व', 'ा', 'ल', 'ं', '林', '彪', '€', '\u200b', '˚', 'ö', '~', '—', '越', '人', 'च', 'म', 'क', 
# 'ु', 'य', 'ी', 'ê', 'ă', 'ễ', '∞', '抗', '日', '神', '剧', '，', '\uf02d', '–', 'ご', 'め', 'な', 'さ', 'い', 'す', 
# 'み', 'ま', 'せ', 'ん', 'ó', 'è', '£', '¡', 'ś', '≤', '¿', 'λ', '魔', '法', '师', '）', 'ğ', 'ñ', 'ř', '그', '자', '식', '멀', 
# '쩡', '다', '인', '공', '호', '흡', '데', '혀', '밀', '어', '넣', '는', '거', '보', '니', 'ǒ', 'ú', '️', 'ش', 'ه', 'ا', 'د',
# 'ة', 'ل', 'ت', 'َ', 'ع', 'م', 'ّ', 'ق', 'ِ', 'ف', 'ي', 'ب', 'ح', 'ْ', 'ث', '³', '饭', '可', '以', '吃', '话', '不', '讲', 
# '∈', 'ℝ', '爾', '汝', '文', '言', '∀', '禮', 'इ', 'ब', 'छ', 'ड', '़', 'ʒ', '有', '「', '寧', '錯', '殺', '一', '千', '絕', 
# '放', '過', '」', '之', '勢', '㏒', '㏑', 'ू', 'â', 'ω', 'ą', 'ō', '精', '杯', 'í', '生', '懸', '命', 'ਨ', 'ਾ', 'ਮ', 'ੁ', 
# '₁', '₂', 'ϵ', 'ä', 'к', 'ɾ', '\ufeff', 'ã', '©', '\x9d', 'ū', '™', '＝', 'ù', 'ɪ', 'ŋ', 'خ', 'ر', 'س', 'ن', 'ḵ', 'ā']
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

stop_words = ['me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",  'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def clean_text(x):
    x = str(x)

    # for stop in stop_words:
    #     # x = x.replace(stop, f ' {stop} ')
    #     x = x.replace(stop, f'{stop}')
    
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
        
    return x

def split_text(x):
    x = wordninja.split(x)
    return '-'.join(x)

def load_and_prec():
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")

    # train_df = train_df[:100]
    # test_df = test_df[:100]
    
    train_df["question_text"] = train_df["question_text"].str.lower()
    test_df["question_text"] = test_df["question_text"].str.lower()
    
    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
    
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)

    ## split to train and val
    # train_df, val_df = train_test_split(train_df, test_size=0.001, random_state=2018) # hahaha

    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    # val_X = val_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    # val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    train_X_len = np.asarray([len(sent) for sent in train_X])
    test_X_len = np.asarray([len(sent) for sent in test_X])

    train_X = pad_sequences(train_X, maxlen=maxlen)
    # val_X = pad_sequences(val_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values
    # val_y = val_df['target'].values  
    
    #shuffling the data
    # np.random.seed(2018)
    # trn_idx = np.random.permutation(len(train_X))
    # val_idx = np.random.permutation(len(val_X))

    # train_X = train_X[trn_idx]
    # # val_X = val_X[val_idx]
    # train_y = train_y[trn_idx]
    # val_y = val_y[val_idx]    
    
    # return train_X, val_X, test_X, train_y, val_y, tokenizer.word_index
    return train_X, test_X, train_y, tokenizer.word_index, train_X_len, test_X_len

def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

if __name__ == '__main__':
    
    # For loading train & test dataset
    if os.path.exists(DATA_DIR + TRAIN_DATA):
        print("test data exists")
        input_data = np.load(open(DATA_DIR + TRAIN_DATA, 'rb'))
        label_data = np.load(open(DATA_DIR + TRAIN_LABEL_DATA, 'rb'))
        data_configs = json.load(open(DATA_DIR + DATA_CONFIGS, 'r'))
        word_vocab = data_configs['vocab']
    else:
        print("no preprocessing file exists")
        train_X, test_X, train_y, word_index, train_X_len, test_X_len = load_and_prec()
        # train_X, val_X, test_X, train_y, val_y, word_index = load_and_prec()
        # vocab = []
        # for w,k in word_index.items():
        #     vocab.append(w)
        #     if k >= max_features:
        #         break

        # train_X, val_X, test_X, train_y, val_y, word_index = load_and_prec()
            
        data_configs = {}
        data_configs['vocab'] = word_index
        data_configs['vocab_size'] = len(word_index) + 1

        json.dump(data_configs, open(DATA_DIR + DATA_CONFIGS, 'w'))
        np.save(open(DATA_DIR + TEST_DATA, 'wb'), test_X)
        np.save(open(DATA_DIR + TEST_LEN_DATA, 'wb'), test_X_len)

        #save train and test dataset into numpy
        np.save(open(DATA_DIR + TRAIN_DATA, 'wb'), train_X)
        np.save(open(DATA_DIR + TRAIN_LEN_DATA, 'wb'), train_X_len)
        np.save(open(DATA_DIR + TRAIN_LABEL_DATA , 'wb'), train_y)

    print("save glove embedding")
    embedding_matrix_1 = load_glove(word_index)
    np.save(open(DATA_DIR + 'glove.npy', 'wb'), embedding_matrix_1)
    
    print("save fasttext embedding")
    embedding_matrix_2 = load_fasttext(word_index)
    np.save(open(DATA_DIR + 'fasttext.npy', 'wb'), embedding_matrix_2)
    
    print("save para embedding")
    embedding_matrix_3 = load_para(word_index)
    np.save(open(DATA_DIR + 'para.npy', 'wb'), embedding_matrix_3)
    
    # np.save(open(DATA_DIR + 'glove.npy', 'wb'), embedding_matrix_1)

    # train_X, test_X, train_y, word_vocab = load_and_prec()

    # embedding_matrix_1 = load_glove(word_vocab)
    # embedding_matrix_2 = load_fasttext(word_vocab)
    # embedding_matrix_3 = load_para(word_vocab)

    