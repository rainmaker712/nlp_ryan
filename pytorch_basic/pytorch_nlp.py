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

#Creating Tensors
V_data = [1,2,3]
V = torch.Tensor(V_data)
print(V)

#Create matrix
M_data = [[1,2,3], [4,5,6]]
M = torch.Tensor(M_data)
print(M)

# Create 3D tensor of size 2*2*2
T_data = [[[1,2],[3,4]],
          [[5,6],[7,8]]]
T = torch.Tensor(T_data)
print(T)

# Index into V and get a scalar
print(V[0])

# Index into M and get a vector
print(M[0])

# Index into T and get a matrix
print(T[0])

x = torch.randn((3, 4, 5))
print(x)

##Operations with Tensors
x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
z = x + y
print(z)

##Concat
# By default, it concatenates along the first axis (concatenates rows)
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# Concatenate columns:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
# second arg specifies which axis to concat along
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# If your tensors are not compatible, torch will complain.  Uncomment to see the error
# torch.cat([x_1, x_2])


##Reshaping Tensors
x = torch.randn(2,3,4)
print(x)
print(x.view(2,12)) #2rows with 12 col.
print(x.view(2,-1)) #Same, If one of the dim. is -1, its size can be inferred

#Comp. Graphs and Auto Diff: How your data is combeind

# Variables wrap tensor objects
x = autograd.Variable(torch.Tensor([1., 2., 3]), requires_grad=True)
# You can access the data with the .data attribute
print(x.data)

# You can also do all the same operations you did with tensors with Variables.
y = autograd.Variable(torch.Tensor([4., 5., 6]), requires_grad=True)
z = x + y
print(z.data)

# BUT z knows something extra.
#print(z.grad_fn) does not work

s = z.sum()
print(s)
#print(s.grad_fn) does not work

s.backward()
print(x.grad)

##Sumamry

x = torch.randn((2,2))
y = torch.randn((2,2))

z= x + y

var_x = autograd.Variable(x)
var_y = autograd.Variable(y)

var_z = var_x + var_y
print(var_z.grad_fn)

var_z_data = var_z.data  # Get the wrapped Tensor object out of var_z...
new_var_z = autograd.Variable(var_z_data)

print(new_var_z.grad_fn)










