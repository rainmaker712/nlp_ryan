#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:24:52 2017

@author: ryan
"""

"""Pytorch Intro"""

import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

#GPU
dtype = torch.cuda.FloatTensor

##Tensors
x = torch.Tensor(5,3).type(dtype)
x = torch.rand(5,3).type(dtype)
x.size()

##Operations
y = torch.rand(5,3).type(dtype)
print(x+y)

#print(torch.add(x,y))
result = torch.Tensor(5,3).type(dtype)
torch.add(x, y, out=result)
print(result)

#Indexing
print(x[:, 1])

##Numpy Bridge

#Convert torch Tensor to numpy Array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

#Convert numpy array to torch
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

#Cuda Tensors
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y


""" Autograd: Automatic differentiation """

##Variable
# If Variable is not a scala, you need to specify arg. for backward()

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2,2), requires_grad=True).type(dtype)
y = x + 2
print(y)

z = y * y * 3
out = z.mean()
print(z, out)


##Gradients
out.backward()

print(x.grad)

import time
from datetime import timedelta

start_time = time.monotonic()
x = torch.randn(3)
x = Variable(x, requires_grad=True)
y = x*2
while y.data.norm() < 1000000:
    y = y * 2
end_time = time.monotonic()

print(timedelta(seconds=end_time - start_time))

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)



