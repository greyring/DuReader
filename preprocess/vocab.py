import numpy as np

class Vocab(object):
    '''
        1.add_list
        2.filter_tokens_by_cnt
        3.randomly_init_embeddings
        4.convert_to_ids
    '''
    def __init__(self, args):
        self.embed_size = args.embed_size
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.id2token = None
        self.token2id = None
        self.vocab = {}#vocab, num
        self.embeddings = None
        
    
    def size(self):
        return len(self.id2token)
    
    def get_id(self, token):
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]
    
    def get_token(self, idx):
        if idx >= len(self.id2token):
            return self.unk_token
        else:
            return self.id2token[idx]
    
    def add_list(self, tokens):
        for token in tokens:
            if token in self.vocab:
                self.vocab[token]+=1
            else:
                self.vocab[token] = 1
    
    def filter_tokens_by_cnt(self, min_cnt):
        for token in self.vocab.keys():
            if self.vocab[token]<min_cnt:
                del self.vocab[token]
    
    def gen_ids(self):
        self.vocab[self.pad_token] = 1
        self.vocab[self.unk_token] = 1
        self.id2token = [x[0] for x in sorted(self.vocab.items(), key=lambda d: d[1], reverse=True)]
        self.token2id = dict(zip(self.id2token, range(len(self.id2token))))
    
    def randomly_init_embeddings(self):
        self.embeddings = np.random.rand(self.size(), self.embed_size)
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_size])

    def convert_to_ids(self, tokens):
        vec = [self.get_id(label) for label in tokens]
        return vec
    
    def recover_from_ids(self, ids):
        tokens = [self.get_token(id) for id in ids]
        return tokens