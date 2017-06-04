#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:51:43 2017

@author: ryan
"""

'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
https://offbit.github.io/how-to-read/
'''

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import itertools
import numpy as np
import random

from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.models import Sequential, Model


"""
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
"""

import os
import sys

np.random.seed(0)

#---------------Data Load Starts------------------
os.getcwd()
os.chdir('/home/ryan/nlp_ryan/Text_Classification')

corpus_fname = './dataset/question_tokenized.txt'

def get_text(fname):
    with open(fname, encoding='utf-8') as f:
        docs = [doc.replace('\n','').split('\t') for doc in f]
        df_docs = docs #에러로 인한 수정
    
    idx, texts, domain = zip(*df_docs)
    return idx, texts, domain               
                
idx, docs, domain = get_text(corpus_fname)

#path = '/home/ryan/nlp_ryan/soy'
#sys.path.append(path)
#print(sys.path)

"""
tokenizer = Tokenizer()
all_texts = x_train + x_test
tokenizer.fit_on_texts(all_texts)
#print(tokenizer.word_index)

x_train = tokenizer.texts_to_matrix(x_train)
x_test = tokenizer.texts_to_matrix(x_test)

#Label converter
all_labels = y_train + y_test
labels = set(all_labels)
idx2labels = list(labels)
label2idx = dict((v,i) for i, v in enumerate(labels))

Y_train = to_categorical([label2idx[w] for w in y_train])
Y_test = to_categorical([label2idx[w] for w in y_test])
"""

#http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/

#https://www.youtube.com/watch?v=ogrJaOIuBx4
#https://www.youtube.com/watch?v=t5qgjJIBy9g&t=247s

from keras.preprocessing.text import Tokenizer

#nb_words= 3 #extract only top word
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs) #단어에 인덱스
print(tokenizer.word_index)
sent = tokenizer.texts_to_sequences(docs)

#------------------Param. Starts ---------------------

# set parameters:
max_features = 5000
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

#Preprecessing param
# max_document_length 계산하기
max_document_length = 0
for document in sent:
    document_length = len(document)
    if document_length > max_document_length:
        max_document_length = document_length

sequence_length = max_document_length
max_words = 5000

#-----------------Param ends-----------------------

print('Pad sequences (samples x time)')
x = sequence.pad_sequences(sent, maxlen=sequence_length, padding="post", truncating="post")
#x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")
#print('x_train shape:', x_train.shape)
#print('x_test shape:', x_test.shape)

#For Domain, alarm: 0 / device: 1 / music: 2 / weather: 3 / time: 4 / etc (chat): 5
#convert domain into with
def conv_str_list(x):
    df_cont = list(map(lambda i: int(i), x))
    return df_cont

df_domain = conv_str_list(domain)
df_idx = conv_str_list(idx)

#Convert domain to binary
def domain_select(dm):
    dm_select = [i for i in range(len(df_domain)) if df_domain[i] == dm]
    etc_select = [i for i in range(len(df_domain)) if df_domain[i] != dm]
    return dm_select, etc_select
    

#Domain Selector
alarm, etc = domain_select(0)

train_prop = 0.8

def data_split(dataset):
    random.seed(1234)
    df_train = random.sample(dataset, round(train_prop * len(dataset)))
    df_test = [element for element in dataset if element not in df_train]
    return df_train, df_test

train_alarm, test_alarm = data_split(alarm)
train_etc, test_etc = data_split(etc)

def data_set(df1, df2):
    x_t = x[df1 + df2]
    y_t = np.r_[
                [[1,0] for _ in df1] + [[0,1] for _ in df2]]
    return x_t, y_t

x_train, y_train = data_set(train_alarm, train_etc)
x_test, y_test = data_set(test_alarm, test_etc)
print(len(x_train), len(y_train), len(x_test), len(y_test))

#tokenizer.word_counts #단어의 숫자변경
#tokenizer.texts_to_matrix(docs) #One hot으로 변경

#-----------------Data Preprocessing Ends -----------

model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=sequence_length))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_data=(x_test, y_test))















#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.fit(X_train, y=y_train, nb_epoch=1500, verbose=0, validation_split=0.2, shuffle=True)
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
