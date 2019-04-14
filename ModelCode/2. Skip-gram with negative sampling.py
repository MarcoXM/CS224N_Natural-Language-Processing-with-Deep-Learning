##
## for every update in iteration, renewing denominator is difficult
## So a trick to release our computational wordkload is nagative sampling


#import pakages

import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import nltk
from torch.autograd import Variable
import torch.nn.functional as F
import random 
from collections import Counter
import torch.optim as optim
flatten = lambda l :[item for sublist in l for item in sublist] ## make elements flat in the data
random.seed(224)


##showing the version of main packages
print(torch.__version__)
print(nltk.__version__)

##because I am a mac and without GPU....but one day! maybe I will run this could in a laptop with powerful machine
## People should keep thier dreams!!!
USE_CUDA = torch.cuda.is_available()


#Try! maybe you can use cuda but you have not found.
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

def getBatch(batch_size,train_data):
    '''
    @batch_size (int) : how many instances you want to train in a iteration
    @train_data (list of data)
    '''
    random.shuffle(train_data)
    start_index =0
    end_index= batch_size
    while end_index < len(train_data):
        batch_data = train_data[start_index:end_index]
        temp = end_index
        end_index += batch_size
        start_index = temp
        yield batch_data
    if end_index >= len(train_data):
        batch_data =train_data[start_index:]
        yield batch_data

def prepare_seq(seq,word2index):
    index = list(map(lambda w : word2index[w] if word2index.get(w) is not None else word2index['<UNK>'],seq))
    return Variable(LongTensor(index))

def prepare_word(word,word2index):
    return Variable(LongTensor(word2index[word])) if word2index.get(word) is not None else LongTensor([word2index['<UNK>']])


# loading data 
corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:500]
corpus = [[word.lower() for word in sent ] for sent in corpus]

vocab = list(set(flatten(corpus)))
print(len(vocab))

word2index = {'<UNK>' : 0}
for i,v in enumerate(vocab):
    if word2index.get(v) is None:
        word2index[v] = i + 1

index2word = { v:k for k,v in word2index.items() }

## Context and centers
WINDOW_SIZE = 5 
win_pairs = flatten([list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 +1 )) for c in corpus])


train_data = []
for word_pair in win_pairs:
    for i in range(WINDOW_SIZE * 2 + 1):
        if i == WINDOW_SIZE and word_pair[i] == '<DUMMY>':
            continue
        train_data.append((word_pair[WINDOW_SIZE],word_pair[i]))

X_tensor = []
y_tensor = []

for data in train_data:
    X_tensor.append(prepare_word(data[0],word2index).view(1,-1))
    y_tensor.append(prepare_word(data[1],word2index).view(1,-1))

inputData = list(zip(X_tensor,y_tensor))
print(len(inputData))


#Building UnigramDistribution ** 0.75
Z = .001
wordCount = Counter(flatten(corpus))
total_ = sum([c for w,c in wordCount.items()])

unigram_table = []
for word in vocab:
    unigram_table.extend([word] * int(((wordCount[word]/total_)**0.75)/Z))

#Negative_smapling
def negative_sampling(targets,unigram_table, k):
    batch_size = targets.size(0)
    neg_samples =[]## the words pair have never happend!!!
    for i in range(batch_size):
        nsample = []
        target_index = targets[i].data.cpu().tolist()[0] if USE_CUDA else targets[i].data.tolist()[0]
        while len(nsample)< k:
            neg =random.choice(unigram_table)
            if word2index[neg] == target_index:
                continue
            nsample.append(neg)
        neg_samples.append(prepare_seq(nsample,word2index).view(1,-1))

    return torch.cat(neg_samples)


class SkigramNegSampling(nn.Module):

    def __init__(self,vocab_size,word_dim):
        super(SkigramNegSampling,self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, word_dim) # v for center word
        self.embedding_u = nn.Embedding(vocab_size, word_dim) # u for context word
        self.logsigmoid = nn.LogSigmoid()

        ## weight initial
        nn.init.xavier_uniform_(self.embedding_v.weight,gain=1)
        nn.init.xavier_uniform_(self.embedding_u.weight,gain=1)

    def forward(self,center,target,neg):
        center_emb = self.embedding_v(center)
        target_emb = self.embedding_u(target)
        neg_embed = - self.embedding_u(neg)

        pos_score = target_emb.bmm(center_emb.transpose(1,2)).squeeze(2) #B ,1
        neg_score = torch.sum(neg_embed.bmm(center_emb.tarnspose(1,2).squeeze(2),1).view(negs.size(0),-1))

        loss = self.logsigmoid(pos_score) + self.logsigmoid(neg_score)
        return -torch.mean(loss)

    def predict(self,inputs):
        embeds =self.embedding_v(inputs)

        return embeds



#trainning
EMBEDDING_SIZE = 30
BATCH_SIZE = 50
EPOCH = 100
NEG = 10

losses = []
model = SkigramNegSampling(len(word2index),EMBEDDING_SIZE)
if USE_CUDA :
    model = model.cuda()

optimizer = optim.Adam(model.parameters(),lr = .001)

for epoch in range(EPOCH):
    for i,batch in enumerate(getBatch(BATCH_SIZE, inputData)):

        inputs,targets = zip(*batch)
        print(inputs)

        inputs = torch.cat(inputs) # B 1
        targets = torch.cat(targets) # B 1
        negs = negative_sampling(targets,unigram_table,NEG)
        model.zero_grad()

        loss = model(inputs,targets,negs)

        loss.backward()
        optimizer.step()

        losses.append(loss.data.tolist()[0])
    if epoch % 10 == 0:
        print(epoch,np.mean(losses))
        losses = []

        




