from collections import defaultdict
import sys
import numpy as np

class CohesionProbability:
    
    def __init__(self, max_l_length=10, min_count=30):
        self.max_l_length = max_l_length
        self.min_count = min_count
        self.L = {}
        
    def train(self, sents):
        for num_sent, sent in enumerate(sents):
            if num_sent % 5000 == 0:
                sys.stdout.write('\risnerting %d sents... ' % num_sent)
            for token in sent.split():
                for e in range(1, min(self.max_l_length, len(token)) + 1):
                    subword = token[:e]
                    self.L[subword] = self.L.get(subword,0) + 1
        print('\rinserting subwords into L: done')
        print('num subword = %d' % len(self.L))

        self.L = {subword:freq for subword, freq in self.L.items() if freq >= self.min_count}
        print('num subword = %d (after pruning with min count %d)' % (len(self.L), self.min_count))
    
    def get_cohesion(self, word):

        # 글자가 아니거나 공백, 혹은 희귀한 단어인 경우
        if (not word) or ((word in self.L) == False): 
            return 0.0

        if len(word) == 1:
            return 1.0

        word_freq = self.L.get(word, 0)
        base_freq = self.L.get(word[:1], 0)

        if base_freq == 0:
            return 0.0
        else:
            return np.power((word_freq / base_freq), 1 / (len(word) - 1))