#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 00:22:29 2017

@author: ryan
"""

#Check whether Prime number or not

def isPrime(num):
    if num > 0:
        
        if (num % 2) != 0:
            print("{} is prime num".format(num))
        else:
            print("{} is not prime num".format(num))
            
    else:
        print("input value must be greater than zero")
        
a = -3

isPrime(a)