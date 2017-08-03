from collections import defaultdict
import pickle

class Send2Reply:
    
    def __init__(self):
        self.s2r = {}
        self.send_set = []
        self.reply_set = []
        
    def train(self, pairs):
        unique_send_set = defaultdict(lambda: len(unique_send_set))
        unique_reply_set = defaultdict(lambda: len(unique_reply_set))
        
        s2r_ = defaultdict(lambda: defaultdict(lambda: 0))
        for pair in pairs:
            send = pair[0]
            reply = pair[1]
            
            send_idx = unique_send_set[send]
            reply_idx = unique_reply_set[reply]
            
            s2r_[send_idx][reply_idx] += 1
        
        for s, rdict in s2r_.items():
            self.s2r[s] = dict(rdict)
        
        for send, _ in sorted(unique_send_set.items(), key=lambda x:x[1]):
            self.send_set.append(send)
            
        for reply, _ in sorted(unique_reply_set.items(), key=lambda x:x[1]):
            self.reply_set.append(reply)
    
    def get_reply(self, send_idx, n_replies=10):
        replies = self.s2r.get(send_idx, {})
        if not replies:
            return []
        
        replies = sorted(replies.items(), key=lambda x:x[1], reverse=True)
        if n_replies > 0:
            replies = replies[:n_replies]
        
        replies = [(self.reply_set[reply[0]], reply[1]) for reply in replies]
        return replies
    
    def save(self, fname):
        params = {
            's2r': self.s2r, 
            'send_set': self.send_set,
            'reply_set': self.reply_set
        }
        with open(fname, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, fname):
        with open(fname, 'rb') as f:
            params = pickle.load(f)
        self.s2r = params['s2r']
        self.send_set = params['send_set']
        self.reply_set = params['reply_set']