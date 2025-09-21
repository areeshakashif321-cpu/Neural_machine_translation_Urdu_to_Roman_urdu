# app.py
import streamlit as st
import torch
import torch.nn as nn
import json
import math
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu

# ----------------------------
# Load Vocabulary + Merges
# ----------------------------
@st.cache_resource
def load_vocab(vocab_file, merges_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id2token = {idx: tok for tok, idx in vocab.items()}
    token2id = vocab
    merges = []
    with open(merges_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                merges.append((int(parts[0]), int(parts[1])))
    return token2id, id2token, merges

# ----------------------------
# Seq2Seq Model (same as training)
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=n_layers,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0, batch_first=True
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell_cat = torch.cat((cell[-2], cell[-1]), dim=1)
        hidden = self.fc_hidden(hidden_cat).unsqueeze(0)
        cell = self.fc_cell(cell_cat).unsqueeze(0)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim * 2, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        pred = self.fc_out(output.squeeze(1))
        return pred, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dec_layers = decoder.lstm.num_layers

    def forward(self, src, max_len=50):
        batch_size = src.size(0)
        tgt_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(self.device)

        _, (hidden, cell) = self.encoder(src)
        hidden = hidden.repeat(self.dec_layers, 1, 1)
        cell = cell.repeat(self.dec_layers, 1, 1)
        input = torch.tensor([1] * batch_size).to(self.device)  # assume 1 = <SOS>

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            input = output.argmax(1)  # greedy decoding

        return outputs

# ----------------------------
# Utility Functions
# ----------------------------
def decode_ids(ids, id2token):
    return " ".join([id2token.get(i, "<UNK>") for i in ids if i != 0])

def compute_metrics(pred, ref):
    bleu = sentence_bleu([ref.split()], pred.split(), weights=(0.5, 0.5))
    cer = Levenshtein.distance(pred, ref) / max(len(ref), 1)
    return bleu, cer

# ----------------------------
# Streamlit Cover Page
# ----------------------------
st.set_page_config(page_title="Neural Machine Translation: Urdu ↔ Roman Urdu", layout="wide")

st.title("📘 Final Year Project: Neural Machine Translation (Urdu ↔ Roman Urdu)")
st.write("This project implements a **Seq2Seq BiLSTM Encoder–Decoder model** with subword-level tokenization (BPE).")

# Project Info
with st.expander("ℹ️ Project Information"):
    st.markdown("""
    - **Domain:** Natural Language Processing (NLP)  
    - **Goal:** Translate Urdu ↔ Roman Urdu using Neural Machine Translation  
    - **Approach:**  
        - Byte Pair Encoding (BPE) for tokenization  
        - Seq2Seq model with BiLSTM Encoder & LSTM Decoder  
        - Trained on Urdu–Roman Urdu parallel corpus  
    - **Metrics Used:** BLEU, Character Error Rate (CER), Perplexity
    """)

# ----------------------------
# Load Model + Vocab
# ----------------------------
VOCAB_FILE = "parallel_bpe_vocab.json"
MERGES_FILE = "parallel_bpe_merges.txt"
MODEL_PATH = "experiments/experiment_1/model_epoch10.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
token2id, id2token, merges = load_vocab(VOCAB_FILE, MERGES_FILE)
vocab_size = len(token2id)

if "model" not in st.session_state:
    encoder = Encoder(vocab_size, emb_dim=256, hidden_dim=512, n_layers=2, dropout=0.3)
    decoder = Decoder(vocab_size, emb_dim=256, hidden_dim=512, n_layers=4, dropout=0.3)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    st.session_state.model = model

# ----------------------------
# Translation Demo
# ----------------------------
st.header("📝 Try the Translation Demo")
input_ids = st.text_area("Enter tokenized IDs (space-separated):", "61 126 237 18")
ref_text = st.text_area("Enter reference translation (optional):", "")

if st.button("🔄 Translate"):
    ids = list(map(int, input_ids.strip().split()))
    src_tensor = torch.tensor([ids], dtype=torch.long).to(device)

    outputs = st.session_state.model(src_tensor, max_len=50)
    pred_ids = outputs.argmax(-1).squeeze(0).tolist()
    pred_text = decode_ids(pred_ids, id2token)

    st.subheader("🔹 Predicted Translation")
    st.success(pred_text)

    if ref_text.strip():
        bleu, cer = compute_metrics(pred_text, ref_text.strip())
        st.subheader("📊 Evaluation Metrics")
        st.write(f"**BLEU:** {bleu:.4f}")
        st.write(f"**CER:** {cer:.4f}")

# ----------------------------
# Training & Experiments
# ----------------------------
with st.expander("⚙️ Training & Experiments"):
    st.markdown("""
    - **Training–Validation–Test Split:** 50% / 25% / 25%  
    - **Optimizer:** Adam, Loss: Cross-Entropy  
    - **Experiments Conducted:**  
        1. Embedding = 128, Hidden = 256, Dropout = 0.1, LR = 1e-3, Batch = 32  
        2. Embedding = 256, Hidden = 512, Dropout = 0.3, LR = 5e-4, Batch = 64  
        3. Embedding = 512, Hidden = 512, Dropout = 0.5, LR = 1e-4, Batch = 128  
    - **Evaluation Metrics:**  
        - BLEU Score  
        - Character Error Rate (CER)  
        - Perplexity
    """)

st.caption("© 2025 Final Year Project – Neural Machine Translation (Urdu ↔ Roman Urdu)")
