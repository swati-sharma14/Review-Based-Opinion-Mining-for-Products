import torch
import pandas as pd
from .constants import DEVICE,contractions


data = pd.read_csv("./datasets/Product_Reviews_with_Description/reviewsWithSummary.csv")


data = pd.read_json("./datasets/Product_Reviews_with_Description/unique_product_reviews.json",orient='records',lines=True)
data = data.sample(5000)
data.reset_index(drop=True,inplace=True)


data.head()


summaryData = pd.read_csv("./datasets/Reviews_with_Summary/Reviews.csv")


summaryData.head()


import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def cleanText(text):
    splitText = text.split()
    newText = []
    sw = stopwords.words('english')
    for word in splitText:
        if word in sw:
            continue
        if word in contractions:
            newText.append(contractions[word])
        else:
            newText.append(word)
    text = " ".join(newText)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    return text

def cleanTextList(textList):
    newText = []
    for text in textList:
        if text is None:
            continue
        newText.append(cleanText(text))
    return newText

summaryData['Text'] = summaryData['Text'].apply(cleanText)
data['reviewText'] = data['reviewText'].apply(cleanTextList)


data.head()


summaryData.head()


maxlen,avglen = 0,0
for sample in data['reviewText']:
    sample = " ".join(sample)
    maxlen = max(maxlen,len(sample))
    avglen += len(sample)
maxlen,avglen/data.shape[0]


data.shape


from transformers import BartTokenizer,BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
model = BartForConditionalGeneration.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
model.to(DEVICE)

def generateReviewSummary(text):
    text = " ".join(text)
    inps = tokenizer(text,return_tensors="pt",max_length=400,truncation=True,padding="max_length")
    inps.to(DEVICE)
    summaryIDs = model.generate(inps["input_ids"],num_beams=3,min_length=30,max_length=40,early_stopping=True)
    summary = tokenizer.decode(summaryIDs[0],skip_special_tokens=True)
    return summary


data['reviewSummary'] = data['reviewText'].map(generateReviewSummary)


data.to_csv("./datasets/Product_Reviews_with_Description/reviewsWithSummary.csv")


data.head()


import re

def cleanCategory(x):
    return " ".join(x.split("_"))

def cleanText(text):
    text = str(text)
    splitText = text.split()
    newText = []
    sw = stopwords.words('english')
    for word in splitText:
        if word in sw:
            continue
        if word in contractions:
            newText.append(contractions[word])
        else:
            newText.append(word)
    text = " ".join(newText)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    return text

data['category'] = data['category'].apply(cleanCategory)
data['description'] = data['description'].apply(cleanText)


data['overall'] = data['overall'].apply(cleanText)


data['category'] = data['category'] + " [SEP] " + data['title']
data = data.drop(columns=['title'])


data


data = data[~data['description'].str.contains(r"^\s+$")]


data


data.rename(columns={"reviewText":"text","overall":"rating"},inplace=True)
data['label'] = 0
data


data2 = data.drop_duplicates()
data2.rename(columns={'description':'prodDesc'},inplace=True)
data2


from torch import Tensor
from torch.nn import Module,Dropout,Linear,LayerNorm
from torch.utils.data import DataLoader, Dataset


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class CategoryReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.stopwords = stopwords.words('english')
        keep = ['not','no','because','but','against','nor','very']
        for elem in keep:
            self.stopwords.remove(elem)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        txt = str(row["rating"]) + " " + row["text"]
        review = [word for word in txt.split() if word.lower() not in self.stopwords]
        label = row["label"]
        category = row["category"]
        
        reviewEncoding = {}
        reviewEncoding['input_ids'] = review
        if len(reviewEncoding['input_ids'])<511:
            reviewEncoding['input_ids'].append("[EOS]")
            reviewEncoding['input_ids'] += ["[PAD]"]*(511-len(reviewEncoding['input_ids']))
        else:
            reviewEncoding['input_ids'] = reviewEncoding['input_ids'][:510]
            reviewEncoding['input_ids'].append("[EOS]")
        reviewEncoding['category'] = category.split()
        if len(reviewEncoding['category'])<82:
            reviewEncoding['category'] += ["[PAD]"]*(82-len(reviewEncoding['category']))
        else:
            reviewEncoding['category'] = reviewEncoding['category'][:82]
        reviewEncoding["label"] = torch.tensor(label)
        return reviewEncoding


def collate_fn(batch):
    text, categories, labels = [], [], []
    for sample in batch:
        text.append(sample['input_ids'])
        categories.append(sample['category'])
        labels.append(sample["label"])

    return text, categories, torch.stack(labels)


class PositionalEncodings(Module):
    def __init__(self, d_model:int, max_len:int=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x:Tensor):
        return (self.pe[:x.size(1), :]).unsqueeze(0)


from typing import Any,Tuple
import torch.nn.functional as F

class FakeReviewIdentifier(Module):
    def __init__(self,
                 model,
                 embeddingLayer,
                 embSize:int,
                 d_model:int,
                 vocabSize:int,
                 num_classes:int,
                 dropoutRate:float,
                 actOut:Any):
        
        super().__init__()
        self.embedding = embeddingLayer
        self.transform = Linear(embSize,d_model)
        self.positionalEncoding = PositionalEncodings(d_model)
        self.vocabSize = vocabSize

        self.model = model
        self.dropout = Dropout(dropoutRate)
        self.layerNorm = LayerNorm(d_model)

        self.fc1 = Linear(d_model,d_model//2)
        self.fc2 = Linear(d_model//2,d_model//4)
        self.fc3 = Linear(d_model//4,num_classes)

        self.act1 = F.tanh
        self.act2 = actOut

    def forward(self, text, category, attnMask):
        x = self.embedding(text)
        x = x.to(DEVICE)
        x = self.transform(x)
        
        xCategory = self.embedding(category)
        xCategory = xCategory.to(DEVICE)
        xCategory = self.transform(xCategory)
        
        x += self.positionalEncoding(x)
        xCategory += self.positionalEncoding(xCategory)

        xOut = self.model(x,xCategory,attnMask)
        xOut = xOut[:,0,:]
        xOut = xOut.to(torch.float32)
        
        out = self.fc1(xOut)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.fc3(out)
        out = torch.softmax(out,dim=-1,dtype=torch.float32)

        return out


class CategoryAttention(Module):
    def __init__(self,
                 d_model:int=512,
                 n_heads:int=8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model//n_heads

        self.Wk = Linear(d_model,d_model)
        self.Wq = Linear(d_model,d_model)
        self.Wv = Linear(d_model,d_model)
        self.Wo = Linear(d_model,d_model)

    def forward(self,query,key,value,mask=None):
        batch_size = query.size(0)

        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        Q = Q.view(batch_size,-1,self.n_heads,self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size,-1,self.n_heads,self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size,-1,self.n_heads,self.head_dim).permute(0,2,1,3)

        energy = torch.matmul(Q,K.permute(0,1,3,2)) / torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float32))
        if mask is not None:
            energy.masked_fill_(mask==0,float("-1e20"))

        attnWeights = torch.softmax(energy,dim=1)
        out = torch.matmul(attnWeights,V)
        out = out.permute(0,2,1,3).contiguous().view(batch_size,-1,self.d_model)

        out = self.Wo(out)
        return out


class TransformerEncoderLayer(Module):
    def __init__(self,
                 d_model:int=512,
                 n_heads:int=8,
                 dim_feedforward:int=2048,
                 dropout:float=0.1,
                 activation:Any = F.relu_):
        
        super().__init__()
        self.selfAttn = CategoryAttention(d_model,n_heads)
        
        self.fc1 = Linear(d_model,dim_feedforward)
        self.dropout = Dropout(dropout)
        
        self.fc2 = Linear(dim_feedforward,d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        self.activation = activation

    def forward(self,
                sent1:Tensor,
                attnMask:Tensor=None,
                crossAttn:Tensor=None):
        
        attnSet = self.selfAttn(sent1,sent1,sent1,attnMask)
        resConn = self.dropout1(attnSet)
        if crossAttn is not None:
            resConn += self.dropout2(crossAttn)
        normOut = self.norm1(resConn)

        FFNOutp = self.activation(self.fc1(normOut))
        FFNProd = self.fc2(self.dropout(FFNOutp))

        out = resConn + self.dropout3(FFNProd)
        out = self.norm2(out)
        return out


from torch.nn import ModuleList

class TransformerEncoder(Module):
    def __init__(self,
                 encoder_layer:Module,
                 num_layers:int=8,
                 d_model:int=512,
                 n_heads:int=8,
                 dim_feedforward:int=2048,
                 dropout:float=0.1,
                 activation:Any=F.relu_):
        
        super().__init__()
        self.catAttn = CategoryAttention(d_model,n_heads)
        self.layers = ModuleList([encoder_layer(d_model,n_heads,dim_feedforward,dropout,activation) for _ in range(num_layers)])

    def forward(self,
                sent1:Tensor,
                sent2:Tensor,
                mask:Tuple[Tensor]=None):
        out = sent1
        attnCat = self.catAttn(sent1,sent2,sent2,mask[0])
        for layer in self.layers:
            out = layer(out,mask[1],attnCat)
        return out


from gensim.models.fasttext import load_facebook_vectors

def FasttextEmbedding(sentences,model=load_facebook_vectors("cc.en.300.bin")):
    embeds = []
    for sentence in sentences:
        wordEmbeds = []
        for i in range(len(sentence)):
            wordEmbeds.append(torch.tensor(model[sentence[i]]))
        if wordEmbeds:
            sentenceEmbedding = torch.stack(wordEmbeds)
            clsEmbed = torch.mean(sentenceEmbedding,dim=0).unsqueeze_(0)
            if len(sentences[0])==512:
                sentenceEmbedding = torch.cat([clsEmbed,sentenceEmbedding],dim=0)
            embeds.append(sentenceEmbedding)
    return torch.stack(embeds)


from torch.nn import CrossEntropyLoss
import torch.nn as nn

embSize = 300
d_model = 128
n_heads = 16
num_layers = 4
num_classes = 2
vocab_size = len(tokenizer)
dropout_rate = 0.25
dim_feedforward = 256
activation = F.relu_

baseModel = TransformerEncoder(
    TransformerEncoderLayer,
    num_layers,
    d_model,
    n_heads,
    dim_feedforward,
    dropout_rate,
    activation
)

fakeClassifier = FakeReviewIdentifier(baseModel,FasttextEmbedding,embSize,d_model,vocab_size,num_classes,dropout_rate,F.leaky_relu_)
checkpoint = torch.load("./checkpoints/FakeClassifier.pt")
fakeClassifier.load_state_dict(checkpoint.state_dict())
loss_fn = CrossEntropyLoss()
fakeClassifier = fakeClassifier.to(DEVICE)


def generateMask(t1,t2):
    attnMask = []
    for i in range(len(t1)):
        s1 = t1[i]
        curMask = []
        for word in s1:
            if word=="[PAD]":
                curMask.append(0)
            else:
                curMask.append(1)
        s1Mask = torch.tensor(curMask).unsqueeze_(1)

        s2 = t2[i]
        curMask = []
        for word in s2:
            if word=="[PAD]":
                curMask.append(0)
            else:
                curMask.append(1)
        s2Mask = torch.tensor(curMask).unsqueeze_(0)
        attnMask.append(torch.matmul(s1Mask,s2Mask))
    return torch.stack(attnMask).unsqueeze_(1)


dataLoader = DataLoader(CategoryReviewDataset(data2),batch_size=256,shuffle=True,collate_fn=collate_fn)


realReviews = []
realProds = []

for text,category,label in dataLoader:
    label = label.to(DEVICE)
    with torch.set_grad_enabled(False):
        out = fakeClassifier(text, category, attnMask=(generateMask(text,category).to(DEVICE),generateMask(text,text).to(DEVICE)))
        _,preds = torch.max(out,dim=1)
        
        for i in range(len(text)):
            if preds.tolist()[i]==0:
                realReviews.append([text[i]])
                realProds.append([category[i]])


len(realReviews),len(realProds)


from sentence_transformers import SentenceTransformer,util

similarityModel = SentenceTransformer('paraphrase-distilroberta-base-v1')


sample = data2[data2['asin']=='B00QXIIZ3W']
s1 = str(sample['reviewSummary'])
s2 = str(sample['prodDesc'])[9:-30]
s1,s2


sents = [s1,s2]
sentEmbs = similarityModel.encode(sents,convert_to_tensor=True)


cos_sim_matrix = util.pytorch_cos_sim(sentEmbs, sentEmbs)
cos_sim_matrix


