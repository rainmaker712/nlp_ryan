#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 00:08:58 2017

@author: ryan
"""

"""
Bubble Sort
performance: O(n^2)
space complexity O(1)

Procedure:
Loop1
6,5,3,1 / 5,6,3,1 / 5,3,6,1 /5,3,1,6
Loop2
3,5,1,6 / 3,1,5,6 / 3,1,5,6
Loop3
1,3,5,6
"""

import unittest

def bubblesort(alist):
    for i in range(len(alist)-1):
        for j in range(len(alist)-1):
            if alist[j] > alist[j+1]:
                alist[j], alist[j+1] = alist[j+1], alist[j]
    return alist
               
sort = [4,6,1,3,5,2]
bubblesort(sort)
    
class unit_test(unittest.TestCase):
    def test(self):
        self.assertEqual([1, 2, 3, 4, 5, 6], bubblesort([4, 6, 1, 3, 5, 2]))
        self.assertEqual([1, 2, 3, 4, 5, 6], bubblesort([6, 4, 3, 1, 2, 5]))
        self.assertEqual([1, 2, 3, 4, 5, 6], bubblesort([6, 5, 4, 3, 2, 1]))


