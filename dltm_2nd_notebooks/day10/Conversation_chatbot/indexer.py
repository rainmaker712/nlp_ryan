from sklearn.metrics import pairwise_distances
from scipy.io import mmread, mmwrite


class FullSearchIndexer:
    
    def __init__(self, x=None):
        self.x = x
    
    def kneighbors(self, query, n_neighbors=10):
        dist_full = pairwise_distances(self.x, query)
        sorted_dist_full = sorted(enumerate(dist_full), key=lambda x:x[1])
        idx, dist = zip(*sorted_dist_full[:n_neighbors])
        dist = [d[0] for d in dist]
        return dist, idx
    
    def save(self, fname):
        mmwrite(fname, self.x)
        
    def load(self, fname):
        self.x = mmread(fname)