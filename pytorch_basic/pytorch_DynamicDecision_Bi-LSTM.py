#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:47:34 2017
http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
@author: ryan
"""

# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

def to_scalar(var):
    return var.view(-1).data.to.list()[0]

def argmax(vec):
    _, idx = torch.max(vec,1)
    return to_scalar(idx)
    
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w], for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)