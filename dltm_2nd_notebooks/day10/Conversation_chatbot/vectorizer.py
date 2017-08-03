import pickle
from sklearn.feature_extraction.text import CountVectorizer

class Vectorizer:
    
    def __init__(self, tokenizer=lambda x:x.split()):
        self.vectorizer = CountVectorizer(tokenizer=tokenizer)
    
    def fit_transform(self, docs):
        return self.vectorizer.fit_transform(docs)
    
    def fit(self, docs):
        self.vectorizer.fit(docs)
    
    def transform(self, docs):
        return self.vectorizer.transform(docs)
    
    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.vectorizer.vocabulary_, f)
    
    def load(self, fname):
        with open(fname, 'rb') as f:
            self.vectorizer.vocabulary_ = pickle.load(f)
            
    def vocabs(self):
        return self.vectorizer.vocabulary_