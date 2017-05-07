#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:08:54 2017

#Data
 - Kaggle의 Bag of Words Meets Bags of Popcorn: https://www.kaggle.com/c/word2vec-nlp-tutorial/data
 - Standford의 Glove Pre-trained Model 100d: https://nlp.stanford.edu/projects/glove/
#영어 Ref: https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
#Python3, Tensorflow (1.0 이상) 및 Keras 최신버전에서 작성하였습니다.

@author: ryan
"""

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

#os.environ['KERAS_BACKEND']='theano' #theano유저에게

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Dropout, Activation, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

#Read data from files
data_train = pd.read_csv( "/home/ryan/dataset/IMDB_Kaggle/labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
data_test = pd.read_csv( "/home/ryan/dataset/IMDB_Kaggle/testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "/home/ryan/dataset/IMDB_Kaggle/unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

#Verify the num of reviews (100,000 in total)
print('labeled train reivews: {}, labled test reviews: {}, unlabed train reviews: {}'.format(data_train["review"].size, data_test["review"].size, unlabeled_train["review"].size))

# 데이터 전처리 과정
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

texts = []
labels = []

text = review_to_wordlist(str(data_train.review[0]))

for idx in range(data_train.review.shape[0]):
    text = review_to_wordlist(data_train.review[idx])
    texts.append(str(text))
    labels.append(data_train.sentiment[idx])

## Keras를 활용한 텍스트 토크나이징 과정
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index #74217개
print('Found {} unique tokens.'.format(len(word_index)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#labels = to_categorical(np.asarray(labels))
labels = np.asarray(labels)
print('Shape of data tensor: {}', data.shape) #(25000, 1000)
print('Shape of label tensor: {}', labels.shape) #(25000, 2)


#Data를 랜덤으로 추출하여 training 및 testing 셋으로 나눈다.

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('긍정, 부정 리뷰들의 training & Validation set 갯수')
print('Training: {}, Validation: {}'.format((y_train.sum(axis=0)), (y_val.sum(axis=0))))

"""
#scikit_learn의 기능으로 조금 더 손쉽게 나눌 수 있음.
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(data, labels, test_size=0.3, random_state=42)
print('긍정, 부정 리뷰들의 training & Validation set 갯수')
print('Training: {}, Validation: {}'.format((y_train.sum(axis=0)), (y_val.sum(axis=0))))
"""

#Pre-trained 된 Glove 불러오기: wiki 및 common crawl, twitter 등의 데이터를 수집하여 학습
#그 중에, 이번에 사용한 glove.6B.100은 Wiki 2014와 Gigaword5를 사용하였음
#Glove가 쓰인 목적은, 학습 Set에 보유하고 있지 않는 데이터 (텍스트) 유입 시, 랜덤 백터로 변환 해줌
GLOVE_DIR = "/home/ryan/dataset/glove"

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('총 {} 개의 word vectors가 Glove 6B 100d 보유'.format(len(embeddings_index)))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items(): #word_index는 dict 형태의 단어의 집합
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        #보유하지 않은 단어는 embedding index에서 모두 0로 처리
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            #input_length=MAX_SEQUENCE_LENGTH,
                            input_length=maxlen,
                            trainable=True)
"""
Keras 모델링
Simple Ver. CNN: 128 filters with size 5 + max pooling of 5 and 35
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

#Deeper CNN: Yoon Kim의 논문에서 나온 구조를 구현

"""
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_val.shape)

#Simple Version

#sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') #(?, 1000)
#embedded_sequences = embedding_layer(sequence_input) #(?, 1000, 100)
sequence_input = Input(shape=(maxlen,), dtype='int32') #(?, 1000)
embedded_sequences = embedding_layer(sequence_input) #(?, 1000, 100)


l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_drop1 = Dropout(0.2)(l_pool1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_drop1)
#l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_drop2 = Dropout(0.2)(l_pool2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_drop2)
#l_pool3 = MaxPooling1D(34)(l_cov3) #global max pooling
#l_flat = Flatten()(l_pool3)
l_flat = Flatten()(l_cov3)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(1, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print("Simple CNN Model 학습")
model.summary()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=128)

#Complex Ver - Yoon Kim CNN

convs = []
filter_sizes = [3,4,5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    #l_conv = Conv1D(filters=128,filter_length=fsz,activation='relu')(embedded_sequences)
    l_conv = Conv1D(filters=128, kernel_size=fsz, activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)
    
l_merge = Merge(mode='concat', concat_axis=1)(convs)
l_cov1= Conv1D(filters=128, activation='relu', kernel_size=5)(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(filters=128, activation='relu', kernel_size=5)(l_pool1)
l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(1, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - more complex convolutional neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=20, batch_size=50)


#To achieve the best performances, we can 1) fine tune hyper parameters 2) further improve text preprocessing 3) use drop out layer

