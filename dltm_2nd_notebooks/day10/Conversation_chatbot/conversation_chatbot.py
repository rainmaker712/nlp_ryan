class ConversationBot:
    
    def __init__(self, vectorizer, send_indexer, send2reply, score=None, pipeline=None):
        self.pipeline = pipeline
        self.vectorizer = vectorizer
        self.send_indexer = send_indexer
        self.send2reply = send2reply
        self.score = score
    
    def process(self, send, n_similar_sends=5, n_reply=5):
        # Preprocessing
        
        # Tokenization & Vectorization
        send_vector = self.vectorizer.transform([send])[0]
        if send_vector.sum() == 0:
            return '음...'
        
        # Get similar sends from send indexer
        send_dist, send_idxs = self.send_indexer.kneighbors(send_vector, n_similar_sends)
        
        # Get reply candidates
        reply_candidates = {}
        for send_idx in send_idxs:
            rs = self.send2reply.get_reply(send_idx)
            for reply, count in rs:
                reply_candidates[reply] = reply_candidates.get(reply, 0) + count
        
        # Scoring
        if not reply_candidates:
            return '죄송해요 무슨 말인지 모르겠어요'
        
        # Sorting
        replies = sorted(reply_candidates.items(), key=lambda x:x[1], reverse=True)
        if n_reply > 0:
            replies = replies[:n_reply]
            
        return replies
        