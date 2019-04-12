#package import
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim 
import torch.nn.functional as F
import nltk
import numpy as np
import random
from collections import Counter
import pandas as pd
flatten = lambda l: [item for sublist in l for item in sublist]
np.random.seed(224)


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

def prepare_seq(seq,woed2index):
    idxs = list(map(lambda w : word2index[w] if word2index.get(w) is not None else word2index['<UNK>'],seq))
    return Variable(LongTensor(idxs))

def prepare_word(word,word2index):
    return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None else LongTensor([word2index['<UNK>']]))




############################################################################################################
############################################################################################################
#data loading and preprocessing

df = pd.read_csv('/Users/marcowang/Dropbox/Textnews/TextProject/sample_data/news20190406_285.csv')
news = list(df['NewsContent'])
corpus = list(nltk.corpus.gutenberg.sents('shakespeare-hamlet.txt'))
corpus = [[word.lower() for word in sent]for sent in corpus]

## vocabulary 
vocab = list(set(flatten(corpus)))# means unknow
vocab.append('<UNK>')
print(len(vocab))

##word2index
word2index = {'<UNK>' : 0}
for word in vocab:
    if word2index.get(word) is None:
        word2index[word] = len(word2index)

index2word = {v:k for k,v in word2index.items()} # this is to use the index to word 

## finding the words within windows size:
Window_size = 3

win_wordpairs = flatten(list(nltk.ngrams(['<DUMMY>'] * Window_size + c + ['<DUMMY>'] * Window_size, Window_size * 2 + 1)) for c in corpus)
#windows is a list of lists, every element have 7 words ! 3 *2 + 1 center word!

# Getting the training data, seperating the center word X and related context words Y we going to predict.
train_data = []

for data in win_wordpairs:
    for w in range(Window_size *2 +1):
        if w == data[Window_size] or data[w] == '<DUMMY>':
            continue
        train_data.append((data[Window_size],data[w])) # droping the DUMMY word and geting x,y 


# Since we got the pair words, next is to transform it into pytorch tensor
X_tensor, y_tensor = [],[] # keeping our tensor to feed our model 

for x_y in train_data:
    X_tensor.append(prepare_word(x_y[0],word2index).view(1,-1))
    y_tensor.append(prepare_word(x_y[1],word2index).view(1,-1)) # now we get tensor


## This is the tensor data we have right now 
train_tensor =list(zip(X_tensor,y_tensor))

### Modeling 
class Skigram(nn.Module):

    def __init__(self,vocab_size, wordvec_dim):
        super(Skigram,self).__init__()
        '''
        @vocab_size : for embedding part ,how many distints word you have
        @wordvec_dim : how long the vecter you want in output 
        '''
        self.embedding_v = nn.Embedding(vocab_size, wordvec_dim) # center word
        self.embedding_u = nn.Embedding(vocab_size, wordvec_dim) # context word

        ## weight initial
        nn.init.xavier_uniform_(self.embedding_v.weight,gain=1)
        nn.init.xavier_uniform_(self.embedding_u.weight,gain=1)

    
    def forward(self,center_w,target_w,out_w):
        cent_emb = self.embedding_v(center_w) #Batch(len of center_w) * 1 * D(you just set this para bro!)
        targ_emb = self.embedding_u(target_w) #like above
        outt_emb = self.embedding_u(out_w)  #B V D

        numerator = targ_emb.bmm(cent_emb.transpose(1,2)).squeeze(2) # 1,2for 1,D , s(2) for 2 dimension
        denominator = outt_emb.bmm(cent_emb.transpose(1,2)).squeeze(2)
        loss = - torch.mean(torch.log(torch.exp(numerator)/torch.sum(torch.exp(denominator),1).unsqueeze(1)))

        return loss

    def predict(self,inputs):
        output = self.embedding_v(inputs)

        return output


# training
VEC_DIMENSION = 30
BATCH_SIZE = 100
EPOCH = 100

losses = []
model = Skigram(len(word2index),VEC_DIMENSION)
if USE_CUDA:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(),lr = 0.01)

for epoch in range(EPOCH):
    for i,batch in enumerate(getBatch(BATCH_SIZE,train_tensor)):
        inputs, tagets = zip(*batch)

        inputs = torch.cat(inputs) # B 1
        targets = torch.cat(tagets) # B 1
        vocab_tensor = prepare_seq(list(vocab),word2index).expand(inputs.size(0),len(vocab))# B V ,every row is same number
        model.zero_grad()

        loss  = model(inputs,targets,vocab_tensor)

        loss.backward()
        optimizer.step()
        print(loss.data)
        losses.append(loss.data)

    if epoch % 10 == 0:
        print(epoch,np.mean(losses))
        losses = []


##Test by similarity
def wordSimilarity(word,vocab):
    if USE_CUDA:
        vec = model.predict(prepare_word(word,word2index))
    else:
        vec = model.predict(prepare_word(word,word2index))
    simi = []

    for i in range(len(vocab)):
        if vocab[i] == word: continue
        
        if USE_CUDA:
            vec_test = model.predict(prepare_word(list(vocab)[i],word2index))
        else:
            vec_test = model.predict(prepare_word(list(vocab)[i],word2index))
        cosdis = F.cosine_simlarity(vec,vec_test).data.tolist()[0]
        simi.append([list(vocab)[i],cosdis])
    return sorted(simi, key = lambda x: x[1], reverse = True)[:10]

#test 
wordSimilarity('love',vocab)









