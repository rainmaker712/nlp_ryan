#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 11:48:58 2017

@author: ryan
http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

word_to_ix = {"안녕": 0, "반가워": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.LongTensor([word_to_ix["안녕"]])
hello_embed = embeds(autograd.Variable(lookup_tensor))
print(hello_embed)

##
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# We will use Shakespeare Sonnet 2
test_sentence = """미국 로스앤젤레스에 사는 배우 척 매카시는 사람들과 산책을 해주고 돈을 번다. 지난해 그가 시작한 '친구 대여(Rent-a-Friend)'는 새로운 형태의 비즈니스다. 매카시는 일감이 많지 않은 무명 배우였지만 이 부업은 조수들을 고용해야 할 만큼 번창하고 있다. 다른 도시와 외국에서도 '출장 산책' 주문이 쇄도한다.

매카시는 집 근처 공원과 거리를 고객과 함께 걸으면서 이야기를 나누는 대가로 1마일(1.6㎞)에 7달러를 받는다. 사회적 관계를 구매 가능한 상품으로 포장한 셈이다. 이름 붙이자면 '고독 비즈니스'다. 그는 영국 일간지 가디언과의 인터뷰에서 "혼자 산책하기 두렵거나 친구 없는 사람으로 비칠까봐 걱정하는 사람이 많았다"며 "자기 이야기를 누가 들어준다는 데 기뻐하며 다시 나를 찾는다"고 했다.

20~30대에서는 미혼과 만혼(晩婚), 40대 이후로는 이혼과 고령화 등으로 1인 가구가 빠르게 늘어가는 한국 사회에서 고독은 강 건너 불구경이 아니다. 우리는 페이스북·트위터·인스타그램 같은 소셜미디어로 긴밀하게 연결돼 있지만 관계의 응집력은 어느 때보다 느슨하다. '혼밥' '혼술' '혼영(나 홀로 영화)' '혼행(나 홀로 여행)' 같은 소비 패턴이 방증한다. 외로움을 감추기보다 즐기려는 경향도 나타난다. Why?는 예스24에 의뢰해 지난 1~5일 설문조사를 했다. 5864명(여성 4398명)이 응답했다. 고독을 바라보는 한국인의 태도가 드러났다.
""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) -2)]

#중복 단어 제외 및 일반 단어 넣어 주기
vocab = set(test_sentence)
word_to_ix = {word: i for i , word in enumerate(vocab)}

#https://wikidocs.net/28

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs
        
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        
        #Step1: 입력전처리 (integer indices(색인) 와 변수로 변환)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        
        #Step2: torch는 gradients를 accumlates한다. 새로운 instances를 넘기기 전에,
        #모든 그레디언트를 오래된 instnaces로 부터 zero out 해야함
        model.zero_grad()
        
        #Step3: 전진 학습을 하며, 다음 단어에 대한 log prob.얻기
        log_probs = model(context_var)
        
        #Step4: log function 사용하기
        loss = loss_function(log_probs, autograd.Variable(
                                                          torch.LongTensor([word_to_ix[target]])))
        
        #Step5: 백프로게이션 실행 후 그레디언트 수치 업데이트
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
    losses.append(total_loss)
print(losses)


"""Exercise: CBow"""
#.view() check
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []

for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

class CBOW(nn.Module):
    
    def __init__(self):
        pass
    
    def forward(self, inputs):
        pass

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)
    
make_context_vector(data[0][0], word_to_ix)






