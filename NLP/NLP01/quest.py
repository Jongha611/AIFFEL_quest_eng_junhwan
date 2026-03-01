# -*- coding: utf-8 -*-

import os
import re
import urllib.request
import tarfile
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pecab import PeCab
from nltk.translate.bleu_score import corpus_bleu

# -------------------------------------------------------------------------------------------------------
# Step 1 & 2: Data Handling and Preprocessing
# -------------------------------------------------------------------------------------------------------

class DataHandler:
    def __init__(self, data_url, data_filename, ko_path, en_path, cache_path):
        self.data_url = data_url
        self.data_filename = data_filename
        self.ko_path = ko_path
        self.en_path = en_path
        self.cache_path = cache_path
        self.pecab = PeCab()

    def download_data(self):
        print(f"Downloading {self.data_filename}...")
        urllib.request.urlretrieve(self.data_url, self.data_filename)
        print("Download complete.")

    def extract_data(self):
        print(f"Extracting {self.data_filename}...")
        with tarfile.open(self.data_filename, "r:gz") as tar:
            tar.extractall()
        print("Extraction complete.")

    def load_raw_data(self):
        if not os.path.exists(self.ko_path) or not os.path.exists(self.en_path):
            if not os.path.exists(self.data_filename):
                self.download_data()
            self.extract_data()

        with open(self.ko_path, "r", encoding="utf-8") as f:
            ko_corpus = f.read().splitlines()
        with open(self.en_path, "r", encoding="utf-8") as f:
            en_corpus = f.read().splitlines()
            
        # Remove duplicates while keeping pairs consistent
        cleaned_corpus = list(set(zip(ko_corpus, en_corpus)))
        return cleaned_corpus

    def preprocess_sentence(self, sentence, is_ko=False):
        sentence = sentence.lower().strip()
        if is_ko:
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9?.!,]+", " ", sentence)
        else:
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        return sentence.strip()

    def tokenize_corpus(self, corpus, limit=10000):
        if os.path.exists(self.cache_path):
            print(f"Loading cached tokenized corpus from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)

        kor_corpus = []
        eng_corpus = []
        for ko, en in tqdm(corpus[:limit], desc="Preprocessing"):
            ko_pre = self.preprocess_sentence(ko, is_ko=True)
            en_pre = self.preprocess_sentence(en, is_ko=False)
            ko_tokens = self.pecab.morphs(ko_pre)
            en_tokens = ["<start>"] + en_pre.split() + ["<end>"]
            if len(ko_tokens) <= 40 and len(en_tokens) <= 40:
                kor_corpus.append(ko_tokens)
                eng_corpus.append(en_tokens)

        print(f"Saving tokenized corpus to {self.cache_path}...")
        with open(self.cache_path, "wb") as f:
            pickle.dump((kor_corpus, eng_corpus), f)
        return kor_corpus, eng_corpus

    def sequences_to_tensor(self, corpus, tokenizer, max_len=40):
        tensor = []
        for sentence in corpus:
            ids = tokenizer.encode(sentence)
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
            tensor.append(ids)
        return torch.tensor(tensor)

# -------------------------------------------------------------------------------------------------------
# Step 3: Tokenization Classes
# -------------------------------------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<start>", 3: "<end>"}
        self.counts = {}

    def build_vocab(self, corpus):
        for sentence in corpus:
            for word in sentence:
                self.counts[word] = self.counts.get(word, 0) + 1
        sorted_words = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - 4]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence):
        return [self.word2idx.get(word, 1) for word in sentence]

    def decode(self, ids):
        return [self.idx2word.get(id, "<unk>") for id in ids]

class TranslationDataset(Dataset):
    def __init__(self, src_tensor, trg_tensor):
        self.src = src_tensor
        self.trg = trg_tensor
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

# -------------------------------------------------------------------------------------------------------
# Step 4: Model Design
# -------------------------------------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim, vocab_size)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        enc_out = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(a, enc_out).permute(1, 0, 2)
        output, hidden = self.rnn(embedded, hidden)
        output = output.squeeze(0)
        context = context.squeeze(0)
        prediction = self.fc_out(torch.cat((output, context), dim=1))
        return prediction, hidden, a.squeeze(1)

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, max_len=40):
        batch_size = src.shape[1]
        outputs = []
        attentions = []
        enc_output, hidden = self.encoder(src)

        if trg is not None:
            for t in range(trg.shape[0]):
                input = trg[t]
                output, hidden, a = self.decoder(input, hidden, enc_output)
                outputs.append(output.unsqueeze(0))
        else:
            input = torch.full((batch_size,), 2, dtype=torch.long, device=self.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            for t in range(max_len):
                output, hidden, a = self.decoder(input, hidden, enc_output)
                outputs.append(output.unsqueeze(0))
                attentions.append(a.unsqueeze(0))
                input = output.argmax(1)
                finished |= (input == 3)
                if finished.all(): break

        return torch.cat(outputs, dim=0), (torch.cat(attentions, dim=0) if attentions else None)

# -------------------------------------------------------------------------------------------------------
# Step 5: NMT Manager
# -------------------------------------------------------------------------------------------------------

class NMTManager:
    def __init__(self, model, optimizer, criterion, device, data_handler):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.data_handler = data_handler

    def train(self, loader, epochs, kor_tokenizer, eng_tokenizer):
        print("\nStarting Training...")
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for src, trg in progress:
                src, trg = src.permute(1, 0).to(self.device), trg.permute(1, 0).to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(src, trg[:-1, :])
                output = output.view(-1, self.model.decoder.fc_out.out_features)
                trg_label = trg[1:, :].reshape(-1)
                loss = self.criterion(output, trg_label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                progress.set_postfix(loss=loss.item())
            
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f}")
            self.evaluate_sample_cases(kor_tokenizer, eng_tokenizer)

    def translate(self, sentence, kor_tok, eng_tok, max_len=40):
        self.model.eval()
        pre = self.data_handler.preprocess_sentence(sentence, is_ko=True)
        tokens = self.data_handler.pecab.morphs(pre)
        ids = kor_tok.encode(tokens)
        src_tensor = torch.tensor(ids).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            output, _ = self.model(src_tensor, max_len=max_len)
        
        pred_ids = output.argmax(2).squeeze(1).cpu().tolist()
        pred_tokens = eng_tok.decode(pred_ids)
        if "<end>" in pred_tokens:
            pred_tokens = pred_tokens[:pred_tokens.index("<end>")]
        return " ".join(pred_tokens)

    def evaluate_sample_cases(self, kor_tokenizer, eng_tokenizer):
        test_cases = ["오바마는 대통령이다.", "시민들은 도시 속에 산다."]
        for i, tc in enumerate(test_cases):
            print(f"{i+1}) {tc[:4]}: {self.translate(tc, kor_tokenizer, eng_tokenizer)}")

    def calculate_bleu(self, kor_corpus, eng_corpus, kor_tok, eng_tok):
        print("\nCalculating BLEU Score...")
        references = []
        candidates = []

        for kor, eng in tqdm(zip(kor_corpus, eng_corpus), total=len(kor_corpus), desc="Evaluating"):
            # kor and eng are lists of tokens from tokenize_corpus
            # We need the original sentence string for translate method (or tokens if we modify it)
            # Reconstruct sentence for translate method
            kor_sentence = " ".join(kor)
            translation = self.translate(kor_sentence, kor_tok, eng_tok)
            
            # BLEU score needs list of tokens for both candidates and references
            # eng_corpus already has <start> and <end>, we should remove them for BLEU
            ref = [eng[1:-1]] # list of list of tokens
            cand = translation.split()
            
            references.append(ref)
            candidates.append(cand)

        score = corpus_bleu(references, candidates)
        print(f"Final BLEU Score: {score * 100:.2f}")
        return score

# -------------------------------------------------------------------------------------------------------
# Execution Block
# -------------------------------------------------------------------------------------------------------

def run_experiment(config, test_cases):
    # Print Configuration for experiment tracking
    print("\n" + "="*50)
    print("Experiment Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")

    # Data Handling
    handler = DataHandler(
        config["data_url"], 
        config["data_filename"], 
        config["ko_path"], 
        config["en_path"], 
        config["cache_path"]
    )
    raw_data = handler.load_raw_data()
    kor_corpus, eng_corpus = handler.tokenize_corpus(raw_data, limit=len(raw_data))
    
    # Vocabulary & Tensors
    kor_tokenizer = Tokenizer(config["vocab_size"])
    eng_tokenizer = Tokenizer(config["vocab_size"])
    kor_tokenizer.build_vocab(kor_corpus)
    eng_tokenizer.build_vocab(eng_corpus)
    
    # Model Initialization
    encoder = Encoder(config["vocab_size"], config["emb_dim"], config["hid_dim"]).to(config["device"])
    attention = BahdanauAttention(config["hid_dim"]).to(config["device"])
    decoder = Decoder(config["vocab_size"], config["emb_dim"], config["hid_dim"], attention).to(config["device"])
    model = Seq2SeqAttention(encoder, decoder, config["device"]).to(config["device"])

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # NMT Management
    nmt = NMTManager(model, optimizer, criterion, config["device"], handler)
    
    # Split data for evaluation (90% train, 10% test)
    split_idx = int(len(kor_corpus) * 0.9)
    train_kor, test_kor = kor_corpus[:split_idx], kor_corpus[split_idx:]
    train_eng, test_eng = eng_corpus[:split_idx], eng_corpus[split_idx:]
    
    train_kor_tensor = handler.sequences_to_tensor(train_kor, kor_tokenizer)
    train_eng_tensor = handler.sequences_to_tensor(train_eng, eng_tokenizer)
    
    train_dataset = TranslationDataset(train_kor_tensor, train_eng_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    nmt.train(train_loader, config["epochs"], kor_tokenizer, eng_tokenizer)

    # BLEU Evaluation
    nmt.calculate_bleu(test_kor, test_eng, kor_tokenizer, eng_tokenizer)

    # Final Evaluation
    print("\nFinal Evaluation on Sample Cases:")
    for tc in test_cases:
        print(f"Kor: {tc} -> Eng: {nmt.translate(tc, kor_tokenizer, eng_tokenizer)}")

def main():
    # Configuration
    config = {
        "data_url": ("https://github.com/jungyeul/korean-parallel-corpora/raw/master/"
                     "korean-english-news-v1/korean-english-park.train.tar.gz"),
        "data_filename": "korean-english-park.train.tar.gz",
        "ko_path": "korean-english-park.train.ko",
        "en_path": "korean-english-park.train.en",
        "cache_path": "tokenized_corpus.pkl",
        "vocab_size": 12000,
        "emb_dim": 256,
        "hid_dim": 512,
        "epochs": 3,
        "batch_size": 128,
        "lr": 1e-3,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    test_cases = [
        "오바마는 대통령이다.",
        "시민들은 도시 속에 산다.",
        "커피는 필요 없다.",
        "일곱 명의 사망자가 발생했다."
    ]

    # Run the translation experiment
    run_experiment(config, test_cases)

if __name__ == "__main__":
    main()
