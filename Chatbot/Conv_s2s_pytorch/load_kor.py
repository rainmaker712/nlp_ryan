from torchtext.vocab import Vocab
from torchtext import data, datasets
#from konlpy.tag import Twitter
#from konlpy.tag import Mecab
import re
import pandas as pd

#twitter = Twitter()
#mecab = Mecab()

import torch
import re
import os
import unicodedata

from config import MAX_LENGTH, save_dir

#train_file_path = './data/chat.in'

SOS_token = 0
EOS_token = 1
PAD_token = 2

# load dataset

# def read_in_data(data_path):
#     delimiter = "\t"
#     user = []
#     bot = []
#     with open(data_path, mode="rt", encoding="utf-8") as fh:
#         utt = fh.readlines()
        
#         for i, line in enumerate(utt):
#             split_line = line.split(delimiter)
#             #print(utt)
#             query = split_line[0].replace("\"", "").replace("\n", "")
#             answer = split_line[1].replace("\"", "").replace("\n", "")
            
#             user.append(query)
#             bot.append(answer)
            
#         return user, bot

# user_query, answer_query = read_in_data(train_file_path)

# #Just for test
# user_query = user_query[:100]
# answer_query = answer_query[:100]

# data_pairs = [] #질/답 형식으로 번갈아 가면서 자장 시키기

# for i in range(len(user_query)):
#     data_pairs.append(user_query[i])
#     data_pairs.append(answer_query[i])

# with open('./data/chat_test.in', "w", encoding="utf-8") as train:
#     for i in range(len(user_query)):
#         train.writelines(user_query[i] + "\n")
#         train.writelines(answer_query[i] + "\n")

class Voc:
    def __init__(self):
        self.index2word = []
        self.word2index = {}
        self.vocab_size = 0
        #self.tokenizer = tokenizer
        
    def add_word(self, word):
        try:
            assert isinstance(word, str)
            if word not in self.word2index:
                self.index2word.append(word)
                self.word2index[word] = self.vocab_size
                self.vocab_size += 1
            
        except AssertionError:
            print('Input should be str')

    def add_sentence(self, sentence):
        #words = self.tokenizer(sentence)
        for word in sentence:
            self.add_word(word)

    def __len__(self):
        return self.vocab_size


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# def tokenizer(sentence):
#     out_list = []
#     for word, pos in mecab.pos(sentence):
#         out_list.append(word)
#     out_list = ' '.join(out_list)

#     return out_list

def normalizeString(s):
    # Lowercase, trim, and remove non-letter characters
    s = re.sub(r"""[,:;?!_\-'\"\.(){}\[\]/\\]""", r" ", s)
    #s = tokenizer(s) #토크나이저 변경가능
    s = unicodeToAscii(s.lower().strip()) #한글이기 때문에 제외
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # s = re.sub(r"\s+", r" ", s).strip()

    return s

def filterPair(p):
    # input sequences need to preserve the last word for EOS_toke
    # 단순 길이 테스트
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 

def filterPairs(pairs):
    # 필터링 된 페어를 저장시키기
    return [pair for pair in pairs if filterPair(pair)]

def readVocs(corpus, corpus_name):
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    with open(corpus, encoding='utf-8') as f:
        content = f.readlines()
    # import gzip
    # content = gzip.open(corpus, 'rt')
    lines = [x.strip() for x in content]
    it = iter(lines)
    # pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]
    pairs = [[x, next(it)] for x in it]

    voc = Voc()
    # for i in range(len(pairs)):
    #     voc.add_sentence(pairs[i][0])
    #     voc.add_sentence(pairs[i][1])

    #voc = Voc(corpus_name)

    return voc, pairs

def prepareData(corpus, corpus_name):
    #corpus: chat.in corpus_name: chat
    voc, pairs = readVocs(corpus, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for i in range(len(pairs)):
        voc.add_sentence(pairs[i][0])
        voc.add_sentence(pairs[i][1])
    
    directory = os.path.join(save_dir, 'training_data', corpus_name) 
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs

def loadPrepareData(corpus):
    corpus_name = corpus.split('/')[-1].split('.')[0]
    try:
        print("Start loading training data ...")
        voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
    except FileNotFoundError:
        print("Saved data not found, start preparing trianing data ...")
        voc, pairs = prepareData(corpus, corpus_name)
    return voc, pairs

#a, b = loadPrepareData('./data/chat.in')
#a, b = prepareData('./data/chat.in', 'chat')
#vars(a)
#a.word2idx