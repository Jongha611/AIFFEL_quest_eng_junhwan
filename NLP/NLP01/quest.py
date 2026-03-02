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
import sentencepiece as spm
from pecab import PeCab
from nltk.translate.bleu_score import corpus_bleu
from typing import List, Tuple, Dict, Any, Optional

# -------------------------------------------------------------------------------------------------------
# Step 1 & 2: Data Handling and Preprocessing
# -------------------------------------------------------------------------------------------------------

class DataHandler:
    def __init__(self, data_url: str, data_filename: str, ko_path: str, en_path: str):
        self.data_url = data_url
        self.data_filename = data_filename
        self.ko_path = ko_path
        self.en_path = en_path

    def download_data(self) -> None:
        print(f"Downloading {self.data_filename}...")
        urllib.request.urlretrieve(self.data_url, self.data_filename)
        print("Download complete.")

    def extract_data(self) -> None:
        print(f"Extracting {self.data_filename}...")
        with tarfile.open(self.data_filename, "r:gz") as tar:
            tar.extractall()
        print("Extraction complete.")

    def load_raw_data(self) -> List[Tuple[str, str]]:
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

    def preprocess_sentence(self, sentence: str) -> str:
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # Keep Korean, English, and basic punctuation
        sentence = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9?.!,]+", " ", sentence)
        return sentence.strip()

    def prepare_spm_data(self, corpus: List[Tuple[str, str]]) -> Tuple[str, str]:
        ko_corpus_path = "ko_spm_corpus.txt"
        en_corpus_path = "en_spm_corpus.txt"
        
        with open(ko_corpus_path, "w", encoding="utf-8") as f_ko, \
             open(en_corpus_path, "w", encoding="utf-8") as f_en:
            for ko, en in corpus:
                f_ko.write(self.preprocess_sentence(ko) + "\n")
                f_en.write(self.preprocess_sentence(en) + "\n")
        
        return ko_corpus_path, en_corpus_path

    def train_sentencepiece(self, input_file: str, model_prefix: str, vocab_size: int) -> None:
        print(f"Training SentencePiece for {model_prefix}...")
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3
        )

    def load_tokenizer(self, model_path: str) -> spm.SentencePieceProcessor:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(model_path)
        return tokenizer

    def sequences_to_tensor(self, sentences: List[str], tokenizer: spm.SentencePieceProcessor, max_len: int = 40) -> torch.Tensor:
        tensor = []
        for sentence in sentences:
            ids = [tokenizer.bos_id()] + tokenizer.encode(sentence) + [tokenizer.eos_id()]
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
            tensor.append(ids)
        return torch.tensor(tensor)

# -------------------------------------------------------------------------------------------------------
# Step 3: Tokenization Classes
# -------------------------------------------------------------------------------------------------------

# Tokenizer is replaced by SentencePieceProcessor

class TranslationDataset(Dataset):
    def __init__(self, src_tensor: torch.Tensor, trg_tensor: torch.Tensor):
        self.src = src_tensor
        self.trg = trg_tensor
    def __len__(self) -> int:
        return len(self.src)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src[idx], self.trg[idx]

# -------------------------------------------------------------------------------------------------------
# Step 4: Model Design
# -------------------------------------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, attention: nn.Module):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim, vocab_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, trg: Optional[torch.Tensor] = None, max_len: int = 40) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, data_handler: DataHandler):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.data_handler = data_handler

    def train(self, loader: DataLoader, epochs: int, kor_tok: spm.SentencePieceProcessor, eng_tok: spm.SentencePieceProcessor) -> None:
        print("\nStarting Training...")
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for src, trg in progress:
                src, trg = src.permute(1, 0).to(self.device), trg.permute(1, 0).to(self.device)
                self.optimizer.zero_grad()
                # Teacher forcing: trg[:-1] is input, trg[1:] is target
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
            self.evaluate_sample_cases(kor_tok, eng_tok)

    def translate(self, sentence: str, kor_tok: spm.SentencePieceProcessor, eng_tok: spm.SentencePieceProcessor, max_len: int = 40) -> str:
        self.model.eval()
        pre = self.data_handler.preprocess_sentence(sentence)
        ids = [kor_tok.bos_id()] + kor_tok.encode(pre) + [kor_tok.eos_id()]
        src_tensor = torch.tensor(ids).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            output, _ = self.model(src_tensor, max_len=max_len)
        
        pred_ids = output.argmax(2).squeeze(1).cpu().tolist()
        # Decode the predicted sequence
        pred_sentence = eng_tok.decode(pred_ids)
        return pred_sentence

    def evaluate_sample_cases(self, kor_tok: spm.SentencePieceProcessor, eng_tok: spm.SentencePieceProcessor) -> None:
        test_cases = ["오바마는 대통령이다.", "시민들은 도시 속에 산다."]
        for i, tc in enumerate(test_cases):
            print(f"{i+1}) {tc[:4]}: {self.translate(tc, kor_tok, eng_tok)}")

    def calculate_bleu(self, test_ko_raw: List[str], test_en_raw: List[str], kor_tok: spm.SentencePieceProcessor, eng_tok: spm.SentencePieceProcessor) -> float:
        print("\nCalculating BLEU Score...")
        references = []
        candidates = []

        for kor, eng in tqdm(zip(test_ko_raw, test_en_raw), total=len(test_ko_raw), desc="Evaluating"):
            translation = self.translate(kor, kor_tok, eng_tok)
            
            # Ground truth: split into tokens for BLEU calculation
            # SentencePiece can return tokens as well
            ref = [eng_tok.encode_as_pieces(self.data_handler.preprocess_sentence(eng))] 
            cand = eng_tok.encode_as_pieces(translation)
            
            references.append(ref)
            candidates.append(cand)

        score = corpus_bleu(references, candidates)
        print(f"Final BLEU Score: {score * 100:.2f}")
        return score

# -------------------------------------------------------------------------------------------------------
# Execution Block
# -------------------------------------------------------------------------------------------------------

def run_experiment(config: Dict[str, Any], test_cases: List[str]) -> None:
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
        config["en_path"]
    )
    raw_data = handler.load_raw_data()
    
    # SentencePiece Training
    ko_txt, en_txt = handler.prepare_spm_data(raw_data)
    handler.train_sentencepiece(ko_txt, "ko_spm", config["vocab_size"])
    handler.train_sentencepiece(en_txt, "en_spm", config["vocab_size"])
    
    # Load Tokenizers
    kor_tok = handler.load_tokenizer("ko_spm.model")
    eng_tok = handler.load_tokenizer("en_spm.model")
    
    # Split raw data for tensors and BLEU
    # We use raw strings for SPM encoding and BLEU evaluation
    split_idx = int(len(raw_data) * 0.9)
    train_data, test_data = raw_data[:split_idx], raw_data[split_idx:]
    
    train_ko_raw = [pair[0] for pair in train_data]
    train_en_raw = [pair[1] for pair in train_data]
    test_ko_raw = [pair[0] for pair in test_data]
    test_en_raw = [pair[1] for pair in test_data]

    # Convert to Tensors
    train_ko_tensor = handler.sequences_to_tensor(train_ko_raw, kor_tok)
    train_en_tensor = handler.sequences_to_tensor(train_en_raw, eng_tok)
    
    # Model Initialization
    encoder = Encoder(config["vocab_size"], config["emb_dim"], config["hid_dim"]).to(config["device"])
    attention = BahdanauAttention(config["hid_dim"]).to(config["device"])
    decoder = Decoder(config["vocab_size"], config["emb_dim"], config["hid_dim"], attention).to(config["device"])
    model = Seq2SeqAttention(encoder, decoder, config["device"]).to(config["device"])

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # NMT Management
    nmt = NMTManager(model, optimizer, criterion, config["device"], handler)
    
    train_dataset = TranslationDataset(train_ko_tensor, train_en_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    nmt.train(train_loader, config["epochs"], kor_tok, eng_tok)

    # BLEU Evaluation
    nmt.calculate_bleu(test_ko_raw, test_en_raw, kor_tok, eng_tok)

    # Final Evaluation
    print("\nFinal Evaluation on Sample Cases:")
    for tc in test_cases:
        print(f"Kor: {tc} -> Eng: {nmt.translate(tc, kor_tok, eng_tok)}")

def main() -> None:
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
