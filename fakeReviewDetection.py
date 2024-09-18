import torch
import pandas as pd
from .constants import DEVICE


data = pd.read_csv("./datasets/Amazon_Reviews_full-product/amazon_reviews.csv")
data = data.sample(frac=0.5)
data.reset_index(drop=True,inplace=True)


data.head()


data["label"] = data['LABEL'].replace({"__label1__":1,"__label2__":0})
data = data.drop(columns=["LABEL"])


data["VERIFIED_PURCHASE"] = data['VERIFIED_PURCHASE'].replace({"N":0,"Y":1})

data["category"] = data["PRODUCT_CATEGORY"] + " [SEP] " + data["PRODUCT_TITLE"]
data = data.drop(columns=["PRODUCT_TITLE","PRODUCT_CATEGORY"])

data["text"] = data["REVIEW_TITLE"] + " [SEP] " + data["REVIEW_TEXT"]
data = data.drop(columns=["REVIEW_TITLE","REVIEW_TEXT"])

data['rating'] = data['RATING']
data = data.drop(columns=['RATING'])


data.head()


import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


DEVICE = torch.device("cuda:7")


from torch import Tensor
from torch.nn import Module,Dropout,Linear,Embedding,LayerNorm,BatchNorm1d
from torch.utils.data import DataLoader, Dataset


torch.autograd.set_detect_anomaly(True)


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
        # rating = row["rating"]
        
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


train_data = data.sample(frac=0.8)
val_data = data.drop(train_data.index)
test_data = val_data.sample(frac=0.5)


from transformers import AutoTokenizer,AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

train_loader = DataLoader(CategoryReviewDataset(train_data), batch_size=256, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(CategoryReviewDataset(val_data), batch_size=256, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(CategoryReviewDataset(test_data), batch_size=256, shuffle=True, collate_fn=collate_fn)


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


from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
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

model = FakeReviewIdentifier(baseModel,FasttextEmbedding,embSize,d_model,vocab_size,num_classes,dropout_rate,F.leaky_relu_)

param_list = list(model.parameters()) + list(baseModel.parameters())
for param in param_list:
    if param.dim()>1:
        nn.init.xavier_uniform_(param)

optimizer = Adam(param_list, lr=5e-4)
loss_fn = CrossEntropyLoss()


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


def epoch(loader, model, loss_fn, optimizer, is_train:bool=False):
    model.train(is_train)
    total_loss = 0
    total_acc = 0
    for text, category, labels in loader:
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(is_train):
            out = model(text, category, attnMask=(generateMask(text,category).to(DEVICE),generateMask(text,text).to(DEVICE)))
            loss = loss_fn(out, labels)
            _,preds = torch.max(out,dim=1)
            acc = (preds==labels).sum().item()/labels.size(0)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_acc += acc
    return total_loss / len(loader), total_acc / len(loader)


torch.cuda.empty_cache()


num_epochs = 100
train_losses, val_losses, train_accs, val_accs = [], [], [], []
patience = 10
min_val_loss = 1e9
bad_epochs = 0

for currEpoch in range(num_epochs):
    model.to(DEVICE)
    
    train_loss,train_acc = epoch(train_loader, model, loss_fn, optimizer, is_train=True)
    val_loss,val_acc = epoch(val_loader, model, loss_fn, optimizer, is_train=False)

    print(f"Epoch {currEpoch+1}")
    print(f"Train Loss: {train_loss:.4f} Train Accuracy: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val Accuracy: {val_acc:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        bad_epochs = 0
    else:
        bad_epochs += 1
    if bad_epochs == patience:
        break


torch.save(model,"checkpoints/FakeClassifier.pt")


cpt = torch.load("checkpoints/FakeClassifier.pt")
newModel = FakeReviewIdentifier(baseModel,FasttextEmbedding,embSize,d_model,vocab_size,num_classes,dropout_rate,F.leaky_relu_)
newModel.load_state_dict(cpt.state_dict())


import matplotlib.pyplot as plt


plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(train_accs, label='Training accuracy')
plt.plot(val_accs, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


