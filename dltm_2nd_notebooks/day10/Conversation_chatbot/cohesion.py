from collections import defaultdict
import math
import sys

import numpy as np


class CohesionProbability:
    
    def __init__(self, left_min_length=1, left_max_length=10, right_min_length=1, right_max_length=6):
        
        self.left_min_length = left_min_length
        self.left_max_length = left_max_length
        self.right_min_length = right_min_length
        self.right_max_length = right_max_length
        
        self.L = defaultdict(int)
        self.R = defaultdict(int)


    def get_cohesion_probability(self, word):
        
        if not word:
            return (0, 0, 0, 0)
        
        word_len = len(word)

        l_freq = 0 if not word in self.L else self.L[word]
        r_freq = 0 if not word in self.R else self.R[word]

        if word_len == 1:
            return (0, 0, l_freq, r_freq)        

        l_cohesion = 0
        r_cohesion = 0
        
        # forward cohesion probability (L)
        if (self.left_min_length <= word_len) and (word_len <= self.left_max_length):
            
            l_sub = word[:self.left_min_length]
            l_sub_freq = 0 if not l_sub in self.L else self.L[l_sub]
            
            if l_sub_freq > 0:
                l_cohesion = np.power( (l_freq / float(l_sub_freq)), (1 / (word_len - len(l_sub) + 1.0)) )
        
        # backward cohesion probability (R)
        if (self.right_min_length <= word_len) and (word_len <= self.right_max_length):
            
            r_sub = word[-1 * self.right_min_length:]
            r_sub_freq = 0 if not r_sub in self.R else self.R[r_sub]
            
            if r_sub_freq > 0:
                r_cohesion = np.power( (r_freq / float(r_sub_freq)), (1 / (word_len - len(r_sub) + 1.0)) )
            
        return (l_cohesion, r_cohesion, l_freq, r_freq)

    
    def get_all_cohesion_probabilities(self):
        
        cp = {}
        words = set(self.L.keys())
        for word in self.R.keys():
            words.add(word)
        
        for word in words:
            cp[word] = self.get_cohesion_probability(word)
            
        return cp
        
        
    def train(self, sents, num_for_pruning = 0, min_count = 5):
        
        for num_sent, sent in enumerate(sents):            
            for word in sent.split():
                
                if not word:
                    continue
                    
                word_len = len(word)
                
                for i in range(self.left_min_length, min(self.left_max_length, word_len)+1):
                    self.L[word[:i]] += 1
                
#                 for i in range(self.right_min_length, min(self.right_max_length, word_len)+1):
                for i in range(self.right_min_length, min(self.right_max_length, word_len)):
                    self.R[word[-i:]] += 1
                    
            if (num_for_pruning > 0) and ( (num_sent + 1) % num_for_pruning == 0):
                self.prune_extreme_case(min_count)
                
        if (num_for_pruning > 0) and ( (num_sent + 1) % num_for_pruning == 0):
                self.prune_extreme_case(min_count)