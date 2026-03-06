#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pecab import PeCab
import math
import urllib.request
import pickle
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. 데이터 전처리 ---

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z가-힣0-9!?,.]", " ", sentence)
    sentence = re.sub(r" {2,}", " ", sentence)
    sentence = sentence.strip()
    return sentence

def load_data(path='data/ChatbotData.csv'):
    if not os.path.exists(path):
        url = 'https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv'
        os.makedirs('data', exist_ok=True)
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path)
    return df['Q'].tolist(), df['A'].tolist()

questions, answers = load_data()
pecab = PeCab()

que_corpus = [pecab.morphs(preprocess_sentence(q)) for q in questions]

vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3, "<sep>": 4}
for tokens in que_corpus:
    for word in tokens:
        if word not in vocab: vocab[word] = len(vocab)

print(f"Vocab Size: {len(vocab)}")

# --- 3. GPT 모델 구성 ---

def positional_encoding(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, (2*(i//2)) / np.float32(d_model))
    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads, self.d_model = num_heads, d_model
        self.depth = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        QK = torch.matmul(Q, K.transpose(-1, -2))
        scaled_qk = QK / math.sqrt(d_k)
        if mask is not None: scaled_qk += (mask * -1e9)
        attentions = F.softmax(scaled_qk, dim=-1)
        return torch.matmul(attentions, V), attentions

    def forward(self, Q, K, V, mask=None):
        bsz = Q.size(0)
        WQ, WK, WV = self.W_q(Q), self.W_k(K), self.W_v(V)
        
        WQ = WQ.view(bsz, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        WK = WK.view(bsz, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        WV = WV.view(bsz, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        
        out, attn = self.scaled_dot_product_attention(WQ, WK, WV, mask)
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, -1, self.d_model)
        return self.linear(out), attn

class GPTLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.do = nn.Dropout(dropout)

    def forward(self, x, mask):
        res = x
        out, _ = self.mha(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        out = self.do(out) + res
        
        res = out
        out = self.do(self.ffn(self.norm2(out))) + res
        return out

class GPTModel(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, vocab_size, pos_len, dropout=0.2):
        super().__init__()
        self.d_model = float(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_encoding", torch.tensor(positional_encoding(pos_len, d_model), dtype=torch.float32))
        self.do = nn.Dropout(dropout)
        self.layers = nn.ModuleList([GPTLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm_final = nn.LayerNorm(d_model, eps=1e-6)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, mask):
        seq_len = x.size(1)
        out = self.embedding(x) * math.sqrt(self.d_model)
        out += self.pos_encoding[:seq_len, :].unsqueeze(0)
        out = self.do(out)
        for layer in self.layers: out = layer(out, mask)
        out = self.norm_final(out)
        return self.fc(out)

def generate_gpt_masks(seq):
    lookahead_mask = torch.triu(torch.ones(seq.shape[1], seq.shape[1]), diagonal=1).unsqueeze(0).unsqueeze(1).to(device)
    padding_mask = (seq == 0).unsqueeze(1).unsqueeze(2).float().to(device)
    return torch.max(padding_mask, lookahead_mask)

model = GPTModel(n_layers=4, d_model=256, n_heads=8, d_ff=512, vocab_size=len(vocab), pos_len=60)
model.to(device)

# --- 4. Pre-training ---

from torch.utils.data import TensorDataset, DataLoader

def pad_sequences(sequences, max_len):
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return torch.tensor(padded, dtype=torch.long)

MAX_LEN = 40
pretrain_vector = []
for q in que_corpus:
    ids = [vocab["<start>"]] + [vocab.get(w, 1) for w in q] + [vocab["<end>"]]
    pretrain_vector.append(ids)

data = pad_sequences(pretrain_vector, MAX_LEN)
dataset = TensorDataset(data)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 10

model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(loader):
        x_batch = batch[0].to(device)
        x, y = x_batch[:, :-1], x_batch[:, 1:]
        mask = generate_gpt_masks(x)
        
        optimizer.zero_grad()
        output = model(x, mask)
        loss = F.cross_entropy(output.reshape(-1, output.size(-1)), y.reshape(-1), ignore_index=0)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(loader):.4f}")

# --- 5. 결과 확인 ---

def generate_text(model, text, max_len=20):
    model.eval()
    tokens = pecab.morphs(preprocess_sentence(text))
    ids = [vocab["<start>"]] + [vocab.get(w, 1) for w in tokens]
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    
    rev_vocab = {v: k for k, v in vocab.items()}
    
    with torch.no_grad():
        for _ in range(max_len):
            mask = generate_gpt_masks(input_ids)
            logits = model(input_ids, mask)
            pred_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if pred_id.item() == vocab["<end>"]: break
            input_ids = torch.cat([input_ids, pred_id], dim=-1)
    
    res_ids = input_ids.squeeze(0).tolist()[len(ids):]
    return " ".join([rev_vocab.get(i, "<unk>") for i in res_ids])

test_sentences = [
    "지루하다", 
    "오늘 일찍", 
    "간만에 여자친구랑", 
    "집에 있는"
]
print("\n# Final GPT Generation Results (Sentence Completion)")
for q in test_sentences:
    print(f"Input Prefix: {q}")
    print(f"Generated: {generate_text(model, q)}")
