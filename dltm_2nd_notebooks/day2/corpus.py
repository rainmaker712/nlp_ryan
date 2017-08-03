class Corpus:
    
    def __init__(self, fname, iter_sent=False):
        self.fname = fname
        self.iter_sent = iter_sent
        self.doc_length = 0
        self.sent_length = 0
        
    def __iter__(self):
        with open(self.fname, encoding='utf-8') as f:
            for doc in f:
                doc = doc.strip()
                if not self.iter_sent:
                    yield doc
                    continue
                for sent in doc.split('  '):
                    yield sent
                    
    def __len__(self):
        if self.iter_sent:
            if self.sent_length == 0:
                with open(self.fname, encoding='utf-8') as f:
                    for doc in f:
                        self.sent_length += len(doc.strip().split('  '))
            return self.sent_length
        else:
            if self.doc_length == 0:
                with open(self.fname, encoding='utf-8') as f:
                    for num_doc, doc in enumerate(f):
                        continue
                    self.doc_length = (num_doc + 1)
            return self.doc_length