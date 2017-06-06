#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:37:50 2017

@author: ryan
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lin = nn.Linear(5,3)
data = autograd.Variable(torch.randn(2,5))
#차원수 변환
print(lin(data))

#Non linearity
print(data)
print(F.relu(data))

# Softmax is also in torch.functional
data = autograd.Variable(torch.randn(5))
print(data)
print(F.softmax(data))
print(F.softmax(data).sum())  # Sums to 1 because it is a distribution!
print(F.log_softmax(data))  # theres also log_softmax

#BOW 모델 연습

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector 
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2

class BoWClassifier(nn.Module): #inheriting from nn.Module!
    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()
        """
        상속하게 되면 명확히 상속된 클래스 이름을 한정자로 부모 클래스의 속성과
        메소드를 접근 할 수 있지만 super()를 이용하여 부모 클래스를 접근 가능
        Super는 하나의 클래스이다.
        Super를 지정하고 접근하면 클래스의 속성과 메소드를 접근해서 처리 가능
        주로 오버라이딩을 작성할 때 super를 이용하여 상위 속성이나 메소드를 참조
        """
        
        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here








 

