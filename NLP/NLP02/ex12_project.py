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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import pickle
import gensim
import math

import urllib.request

# 한글 폰트 설정 (UserWarning 방지)
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# pip install konlpy
# pip install nltk
# pip install gensim

# ------------------------------------------------------------------------
# 1. 데이터 다운로드 및 로드
# ------------------------------------------------------------------------
DATA_PATH = 'data/ChatbotData.csv'
WV_PATH = 'data/ko.bin'
DATA_URL = 'https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv'
WV_URL = 'https://github.com/Kyubyong/wordvectors/raw/master/ko/ko.bin'

def load_data(path):
    # 파일이 없으면 다운로드 수행
    if not os.path.exists(path):
        print(f"{path} not found. Downloading from {DATA_URL}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
        print("Download completed.")
    
    df = pd.read_csv(path)
    questions = df['Q'].tolist()
    answers = df['A'].tolist()
    return questions, answers

# ------------------------------------------------------------------------
# 2. 데이터 정제
# ------------------------------------------------------------------------
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z가-힣0-9!?,.]", " ", sentence)
    sentence = re.sub(r" {2,}", " ", sentence)
    sentence = sentence.strip()
    return sentence

# ------------------------------------------------------------------------
# 3. 토큰화 및 데이터셋 구축
# ------------------------------------------------------------------------
CORPUS_PATH = 'data/corpus.pkl'

def build_corpus(questions, answers, max_len=40, corpus_path=CORPUS_PATH):
    # 캐시된 파일이 있으면 로드
    if os.path.exists(corpus_path):
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'rb') as f:
            que_corpus, ans_corpus = pickle.load(f)
        return que_corpus, ans_corpus

    print("Building new corpus...")
    pecab = PeCab()
    que_corpus, ans_corpus = [], []
    seen = set()
    
    for q, a in tqdm(zip(questions, answers), total=len(questions), desc="Building Corpus"):
        q = preprocess_sentence(q)
        a = preprocess_sentence(a)
        if (q, a) in seen: continue
        seen.add((q, a))
        
        q_tokens = pecab.morphs(q)
        a_tokens = pecab.morphs(a)
        
        if len(q_tokens) <= max_len and len(a_tokens) <= max_len:
            que_corpus.append(q_tokens)
            ans_corpus.append(a_tokens)
            
    # 로컬에 저장
    print(f"Saving corpus to {corpus_path}...")
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    with open(corpus_path, 'wb') as f:
        pickle.dump((que_corpus, ans_corpus), f)
        
    return que_corpus, ans_corpus

# ------------------------------------------------------------------------
# 4. Augmentation (Lexical Substitution)
# ------------------------------------------------------------------------
def load_word2vec(path):
    # 파일이 없으면 다운로드 수행
    if not os.path.exists(path):
        print(f"{path} not found. Downloading from {WV_URL}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            urllib.request.urlretrieve(WV_URL, path)
            print("Download completed.")
        except Exception as e:
            print(f"Failed to download Word2Vec: {e}")
            return None

    # Try multiple loading methods for older/binary ko.bin files
    try:
        # 1. Native gensim load (handles newer pickles)
        model = gensim.models.Word2Vec.load(path)
        return model.wv
    except:
        try:
            # 2. KeyedVectors load
            return gensim.models.KeyedVectors.load(path)
        except:
            try:
                # 3. Direct pickle load with latin1 (for legacy Python 2 models)
                with open(path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                # Check if it's a Word2Vec model and extract the vectors (syn0 in old versions)
                if hasattr(data, 'syn0') and hasattr(data, 'index2word'):
                    # Create a mock KeyedVectors object
                    from gensim.models.keyedvectors import KeyedVectors
                    kv = KeyedVectors(vector_size=data.vector_size if hasattr(data, 'vector_size') else data.syn0.shape[1])
                    kv.add_vectors(data.index2word, data.syn0)
                    return kv
                return None
            except Exception as e:
                try:
                    # 4. load_word2vec_format (C-binary format)
                    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
                except Exception as e2:
                    print(f"Failed to load Word2Vec: {e2}")
                    return None

def lexical_sub(tokens, wv):
    if not tokens or wv is None: return tokens
    valid_tokens = [tok for tok in tokens if tok in wv]
    if not valid_tokens: return tokens
    
    selected_tok = random.choice(valid_tokens)
    similar_word = wv.most_similar(selected_tok)[0][0]
    return [similar_word if tok == selected_tok else tok for tok in tokens]

def augment_data(que_corpus, ans_corpus, wv):
    augmented_que, augmented_ans = [], []
    
    if wv is None:
        print("Word2Vec model not found. Skipping augmentation and using original data.")
        return que_corpus, ans_corpus

    for q, a in tqdm(zip(que_corpus, ans_corpus), total=len(que_corpus), desc="Augmenting Data"):
        augmented_que.append(q)
        augmented_ans.append(a)
        augmented_que.append(lexical_sub(q, wv))
        augmented_ans.append(a)
        augmented_que.append(q)
        augmented_ans.append(lexical_sub(a, wv))
    return augmented_que, augmented_ans

# ------------------------------------------------------------------------
# 5. 데이터 벡터화
# ------------------------------------------------------------------------
def tokenize_and_vectorize(que_corpus, ans_corpus):
    vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
    for sentence in que_corpus + ans_corpus:
        for word in sentence:
            if word not in vocab: vocab[word] = len(vocab)
                
    def sentence_to_ids(sentence, vocab, is_target=False):
        ids = [vocab["<start>"]] if is_target else []
        ids.extend([vocab.get(word, vocab["<unk>"]) for word in sentence])
        if is_target: ids.append(vocab["<end>"])
        return ids

    que_vector = [sentence_to_ids(s, vocab) for s in que_corpus]
    ans_vector = [sentence_to_ids(s, vocab, is_target=True) for s in ans_corpus]
    return que_vector, ans_vector, vocab

def pad_sequences(sequences, max_len, pad_value=0):
    padded = np.full((len(sequences), max_len), pad_value)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return torch.tensor(padded, dtype=torch.long)

# ------------------------------------------------------------------------
# 6. Transformer Model Components
# ------------------------------------------------------------------------
def positional_encoding(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, (2*(i//2)) / np.float32(d_model))
    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table

def generate_padding_mask(seq):
    return (seq == 0).unsqueeze(1).unsqueeze(2).float()

def generate_lookahead_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1)

def generate_masks(src, tgt, device):
    enc_mask = generate_padding_mask(src).to(device)
    dec_enc_mask = generate_padding_mask(src).to(device)
    dec_lookahead_mask = generate_lookahead_mask(tgt.shape[1]).unsqueeze(0).unsqueeze(1).to(device)
    dec_tgt_padding_mask = generate_padding_mask(tgt).to(device)
    dec_mask = torch.max(dec_tgt_padding_mask, dec_lookahead_mask)
    return enc_mask, dec_enc_mask, dec_mask

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

    def split_heads(self, x):
        bsz, seq_len, _ = x.size()
        return x.view(bsz, seq_len, self.num_heads, self.depth).permute(0, 2, 1, 3)

    def combine_heads(self, x):
        bsz, num_heads, seq_len, depth = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        WQ, WK, WV = self.W_q(Q), self.W_k(K), self.W_v(V)
        out, attn = self.scaled_dot_product_attention(self.split_heads(WQ), self.split_heads(WK), self.split_heads(WV), mask)
        return self.linear(self.combine_heads(out)), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.norm1, self.norm2 = nn.LayerNorm(d_model, eps=1e-6), nn.LayerNorm(d_model, eps=1e-6)
        self.do = nn.Dropout(dropout)
    def forward(self, x, mask):
        res = x
        out, attn = self.mha(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        out = self.do(out) + res
        res = out
        out = self.do(self.ffn(self.norm2(out))) + res
        return out, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model, eps=1e-6), nn.LayerNorm(d_model, eps=1e-6), nn.LayerNorm(d_model, eps=1e-6)
        self.do = nn.Dropout(dropout)
    def forward(self, x, enc_out, dec_enc_mask, padding_mask):
        res = x
        out, attn1 = self.mha1(self.norm1(x), self.norm1(x), self.norm1(x), padding_mask)
        out = self.do(out) + res
        res = out
        out, attn2 = self.mha2(self.norm2(out), enc_out, enc_out, dec_enc_mask)
        out = self.do(out) + res
        res = out
        out = self.do(self.ffn(self.norm3(out))) + res
        return out, attn1, attn2

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
    def forward(self, x, mask):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attns.append(attn)
        return x, attns

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
    def forward(self, x, enc_out, dec_enc_mask, padding_mask):
        attns, enc_attns = [], []
        for layer in self.layers:
            x, attn, enc_attn = layer(x, enc_out, dec_enc_mask, padding_mask)
            attns.append(attn)
            enc_attns.append(enc_attn)
        return x, attns, enc_attns

class Transformer(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, src_vocab_size, tgt_vocab_size, pos_len, dropout=0.2):
        super().__init__()
        self.d_model = float(d_model)
        self.enc_emb = nn.Embedding(src_vocab_size, d_model)
        self.dec_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.register_buffer("pos_encoding", torch.tensor(positional_encoding(pos_len, d_model), dtype=torch.float32))
        self.do = nn.Dropout(dropout)
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
    def embedding(self, emb, x):
        seq_len = x.size(1)
        out = emb(x) * math.sqrt(self.d_model)
        out += self.pos_encoding[:seq_len, :].unsqueeze(0)
        return self.do(out)
    def forward(self, enc_in, dec_in, enc_mask, dec_enc_mask, dec_mask):
        enc_out, _ = self.encoder(self.embedding(self.enc_emb, enc_in), enc_mask)
        dec_out, _, _ = self.decoder(self.embedding(self.dec_emb, dec_in), enc_out, dec_enc_mask, dec_mask)
        return self.fc(dec_out)

# ------------------------------------------------------------------------
# 7. Training & Evaluation Logic
# ------------------------------------------------------------------------
def loss_function(real, pred):
    loss_ = F.cross_entropy(pred.contiguous().view(-1, pred.size(-1)), real.contiguous().view(-1), reduction='none')
    mask = (real != 0).float()
    return (loss_.view(real.size()) * mask).sum() / mask.sum()

class LearningRateScheduler:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model, self.warmup_steps = d_model, warmup_steps
    def __call__(self, step):
        step = float(step + 1)
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

def calculate_bleu(reference, candidate):
    return sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)

def evaluate(sentence, transformer, vocab, rev_vocab, device, max_len=40):
    transformer.eval()
    sentence = preprocess_sentence(sentence)
    pecab = PeCab()
    tokens = pecab.morphs(sentence)
    
    enc_in = torch.tensor([[vocab.get(t, vocab["<unk>"]) for t in tokens]], dtype=torch.long).to(device)
    dec_in = torch.tensor([[vocab["<start>"]]], dtype=torch.long).to(device)
    
    for _ in range(max_len):
        enc_mask, dec_enc_mask, dec_mask = generate_masks(enc_in, dec_in, device)
        predictions = transformer(enc_in, dec_in, enc_mask, dec_enc_mask, dec_mask)
        prediction = predictions[:, -1:, :].argmax(dim=-1)
        
        if prediction.item() == vocab["<end>"]: break
        dec_in = torch.cat([dec_in, prediction], dim=-1)
        
    ids = dec_in.squeeze().tolist()[1:]
    return [rev_vocab[i] for i in ids]

def visualize_attention(sentence, response, attention, tokens):
    sentence = preprocess_sentence(sentence)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    
    # attention: [1, num_heads, tgt_len, src_len]
    if len(attention.shape) == 4:
        attention = attention.squeeze(0)
        
    # attention: [num_heads, tgt_len, src_len]
    # We take the average over heads
    attention = attention.mean(dim=0).cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='viridis')
    fig.colorbar(cax)
    
    # 눈금 위치를 먼저 설정하여 UserWarning 방지
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(response)))
    
    # 라벨 설정
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(response)
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    filename = f"results/attention_{re.sub(r'[^a-zA-Z가-힣0-9]', '_', sentence)[:20]}.png"
    plt.savefig(filename)
    print(f"Attention plot saved to {filename}")
    
    plt.show()

def evaluate_with_attention(sentence, transformer, vocab, rev_vocab, device, max_len=40):
    transformer.eval()
    sentence = preprocess_sentence(sentence)
    pecab = PeCab()
    tokens = pecab.morphs(sentence)
    
    enc_in = torch.tensor([[vocab.get(t, vocab["<unk>"]) for t in tokens]], dtype=torch.long).to(device)
    dec_in = torch.tensor([[vocab["<start>"]]], dtype=torch.long).to(device)
    
    attention_weights = None
    
    for _ in range(max_len):
        enc_mask, dec_enc_mask, dec_mask = generate_masks(enc_in, dec_in, device)
        # To get attention weights, we might need a slightly modified forward or a hook
        # For simplicity, if the Transformer class doesn't return them, we only get the output.
        # I will modify the Transformer class forward to return attention weights.
        predictions, enc_out, dec_enc_attns = transformer_with_attn(transformer, enc_in, dec_in, enc_mask, dec_enc_mask, dec_mask)
        prediction = predictions[:, -1:, :].argmax(dim=-1)
        
        if prediction.item() == vocab["<end>"]: break
        dec_in = torch.cat([dec_in, prediction], dim=-1)
        attention_weights = dec_enc_attns[-1] # Take last layer attention
        
    ids = dec_in.squeeze().tolist()[1:]
    response = [rev_vocab[i] for i in ids]
    return response, attention_weights, tokens

def transformer_with_attn(model, enc_in, dec_in, enc_mask, dec_enc_mask, dec_mask):
    enc_out, _ = model.encoder(model.embedding(model.enc_emb, enc_in), enc_mask)
    dec_out, dec_attns, dec_enc_attns = model.decoder(model.embedding(model.dec_emb, dec_in), enc_out, dec_enc_mask, dec_mask)
    logits = model.fc(dec_out)
    return logits, enc_out, dec_enc_attns

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Data Loading
    print("Step 1: Loading Data...")
    questions, answers = load_data(DATA_PATH)
    
    # Step 2, 3: Tokenization & Filtering
    print("Step 2, 3: Building Corpus...")
    que_corpus, ans_corpus = build_corpus(questions, answers)
    
    # Step 4: Augmentation (Lexical Substitution) : 11749 -> 35247
    print("Step 4: Augmenting Data...")
    wv = load_word2vec(WV_PATH)
    aug_que, aug_ans = augment_data(que_corpus, ans_corpus, wv)
    print(f"Total samples after augmentation: {len(que_corpus)} -> {len(aug_que)}")
    
    # Step 5: Vectorization
    print("Step 5: Vectorizing...")
    que_vector, ans_vector, vocab = tokenize_and_vectorize(aug_que, aug_ans)
    rev_vocab = {v: k for k, v in vocab.items()}
    
    MAX_LEN = 40
    enc_train = pad_sequences(que_vector, MAX_LEN)
    dec_train = pad_sequences(ans_vector, MAX_LEN)
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(enc_train, dec_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Step 6: Model Training
    print("Step 6: Initializing Model...")
    D_MODEL = 368
    transformer = Transformer(n_layers=1, d_model=D_MODEL, n_heads=8, d_ff=1024, 
                              src_vocab_size=len(vocab), tgt_vocab_size=len(vocab), pos_len=MAX_LEN).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LearningRateScheduler(D_MODEL)
    
    EPOCHS = 10
    total_step = 0
    for epoch in range(EPOCHS):
        transformer.train()
        total_loss = 0
        for b_enc, b_dec in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            b_enc, b_dec = b_enc.to(device), b_dec.to(device)
            dec_input, dec_real = b_dec[:, :-1], b_dec[:, 1:]
            enc_mask, dec_enc_mask, dec_mask = generate_masks(b_enc, dec_input, device)
            
            optimizer.param_groups[0]['lr'] = lr_scheduler(total_step)
            optimizer.zero_grad()
            pred = transformer(b_enc, dec_input, enc_mask, dec_enc_mask, dec_mask)
            loss = loss_function(dec_real, pred)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_step += 1
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

    # Step 7: Verification
    print("\nStep 7: Verification...")
    
    # BLEU 측정을 위해 질문-답변 매핑 생성
    qa_dict = {q: a for q, a in zip(questions, answers)}
    
    test_sentences = ["지루해", "오늘 점심 뭐 먹지?", "나랑 놀자", "결정 장애가 있는 것 같아"]
    
    for ts in test_sentences:
        response, attention, tokens = evaluate_with_attention(ts, transformer, vocab, rev_vocab, device)
        print(f"\nQ: {ts}")
        print(f"A: {' '.join(response)}")
        
        # 데이터셋에 원본 질문이 있는 경우 BLEU 스코어 계산
        if ts in qa_dict:
            reference = build_corpus([ts], [qa_dict[ts]])[1][0] # 정답 토큰화
            score = calculate_bleu(reference, response)
            print(f"BLEU Score: {score:.4f}")
        
        # Attention 시각화 
        visualize_attention(ts, response, attention, tokens)
