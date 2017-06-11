#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 18:06:39 2017
# Author: Robert Guthrie
http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3,3) #Input dim, output dim (3,3)
inputs = [autograd.Variable(torch.randn((1, 3)))
          for _ in range(5)]  # make a sequence of length 5

#hidden state 초기화
hidden = (autograd.Variable(torch.randn(1,1,3)),
          autograd.Variable(torch.randn(1,1,3)))

for i in inputs:
    # Step through the sequence one elements at a time.
    # after each step, hidden contains the hidden state
    out, hidden = lstm(i.view(1,1,-1), hidden)
    
# 전체 seq.를 한번에 진행이 가능하다.
# LSTM에서 받은 첫번째 값은 
# 두번째는 가장 최근의 hidden state이다.
# 그 이유는, "out"은 모든 hidden state 차례대로 접근 할 수 있고,
# "hidden"은 seq를 진행하며 backprop을 하게 해주기 때문이다.
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(
          torch.randn(1,1,3)))
out. hidden = lstm(inputs, hidden)
print(out)
print(hidden)

"""LSTM for POS Tagging

"""

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

#일반적으로 약 32~64 차원이지만, 값을 적게하여 학습이 진행 되면 값이 어떻게 보내는지 체크
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

#Create the Model
class LSTMTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        #LSTM -> input: word embeddings / output: hidden state / dim: hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        #linear layer는 hidden에서 tag공간으로 변경
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # The axes semantics are (num_layers, mini_batch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1,1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1,1, self.hidden_dim)))
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
                                          embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        tag_scores = F.log_softmax(tag_space)
        return tag_scores
        
#Training Model
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

#학습 전에 성능을 확인해보자 - i: word / j: tag
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300): #toy data이기 때문에 300번만 하는 것, 원래는 그 이상
    for sentence, tags in training_data:
        #Step1: Pytorch는 gradient를 중첩하는 방식이므로, 각각의 instance들을 명확히 해주는 작업이 필요.
        model.zero_grad()
        
        #또한, hidden state LSTM을 명확히 해주는 것이 필요
        #지난 history를 보유하고 있는 instance를 떼어 정보를 공유
        model.hidden = model.init_hidden()
        
        #Step2: input에서 단어의 index형태로 변환시키는 작업
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        
        #Step3: Run our forward pass.
        tag_scores = model(sentence_in)
        
        #Step4: Compare the loss, gradients, and update the param. by calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        
#학습 후 점수 확인하기
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)
#결과 값을 보면, 예측한 seq는 0 1 2 0 1 (가장 높은 수) 이다.
#문장은 "the dog ate the apple."
#확인해보면, DET, NOUN, VERB, DET, NOUN 이므로 정확한 문장

    

