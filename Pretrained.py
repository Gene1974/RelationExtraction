import numpy as np
import torch

from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath
from Utils import logger
        
def GloveEmbedding(path = None, word_list = None, PAD_TAG = '<PAD>', OOV_TAG = '<OOV>'):
    if path is None:
        #path = '/data/Glove/glove.6B.100d.txt'
        path = '/data/Glove/glove.6B.300d.txt'
    word_to_ix = {PAD_TAG: 0, OOV_TAG: 1}
    word_emb = []
    with open(path, 'r') as glove:
        for line in glove.readlines():
            data = line.strip().split(' ') # [word emb1 emb2 ... emb n]
            word = data[0]
            embeds = [float(i) for i in data[1:]]
            word_to_ix[word] = len(word_to_ix)
            word_emb.append(embeds)
    
    word_emb.insert(0, [0.] * len(word_emb[0]))
    word_emb.insert(0, [0.] * len(word_emb[0]))
    
    if word_list is None:
        word_emb = torch.tensor(word_emb, dtype = torch.float)
    else:
        used_idx = [word_to_ix[word] if word in word_to_ix else word_to_ix[OOV_TAG] for word in word_list]
        word_emb = torch.tensor(word_emb, dtype = torch.float)[used_idx]
    logger('Load Glove embedding: {}'.format(word_emb.shape))
    return word_emb
    
def SennaEmbedding(path = None, word_list = None, PAD_TAG = '<PAD>', OOV_TAG = '<OOV>'):
    if path is None:
        path = '/data/Senna/'
    word_to_ix = {PAD_TAG: 0, OOV_TAG: 1}
    word_emb = []
    with open(path + 'word_list.txt', 'r') as f:
        for line in f.readlines():
            word = line.strip()
            word_to_ix[word] = len(word_to_ix)
    with open(path + 'embeddings.txt', 'r') as f:
        for line in f.readlines():
            embeds = line.strip().split(' ')
            embeds = [float(i) for i in embeds]
            word_to_ix[word] = len(word_to_ix)
            word_emb.append(embeds)
    
    word_emb.insert(0, [0.] * len(word_emb[0]))
    word_emb.insert(0, [0.] * len(word_emb[0]))
    
    if word_list is None:
        word_emb = torch.tensor(word_emb, dtype = torch.float)
    else:
        used_idx = [word_to_ix[word] if word in word_to_ix else word_to_ix[OOV_TAG] for word in word_list]
        word_emb = torch.tensor(word_emb, dtype = torch.float)[used_idx]
    logger('Load Senna embedding: {}'.format(word_emb.shape))
    return word_emb

def Word2vecEmbedding(path = None, word_list = [], PAD_TAG = '<PAD>', OOV_TAG = '<OOV>'):
    if path is None:
        path = '/data/Word2vec/GoogleNews-vectors-negative300.bin'
    wv = KeyedVectors.load_word2vec_format(datapath(path), binary=True) # <class 'gensim.models.keyedvectors.KeyedVectors'>
    matrix=np.random.normal(size = (len(word_list) + 1, 300))
    pretrained = 0

    for i in range(len(word_list)):
        word = word_list[i]
        if word in wv:
            matrix[i, :]=wv[word]
            pretrained += 1

    word_emb = torch.tensor(matrix, dtype = torch.float)
    logger('Load Word2vec embedding: {}, pretrained = {}'.format(word_emb.shape, pretrained))
    return word_emb


if __name__ == '__main__':
    Word2vecEmbedding()
