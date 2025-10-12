# ==========================
# Urdu to Roman Urdu Translator (Stable Streamlit Version)
# Last updated: 2025-10-12
# ==========================

import streamlit as st
import os
import sys
import subprocess

# --------------------------
# AUTO-INSTALL REQUIRED PACKAGES (for Streamlit Cloud / Colab)
# --------------------------
REQUIRED_LIBS = ["torch", "pandas", "pickle5"]

for lib in REQUIRED_LIBS:
    try:
        __import__(lib)
    except ImportError:
        with st.spinner(f"Installing missing library: {lib}..."):
            subprocess.run([sys.executable, "-m", "pip", "install", lib])

# After installation, import the libraries
import torch
import torch.nn as nn
import pickle
import pandas as pd
from collections import defaultdict
import re

# --------------------------
# STREAMLIT CONFIGURATION
# --------------------------
st.set_page_config(
    page_title="Urdu → Roman Urdu Translator",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# SAFE CSS
# --------------------------
st.markdown("""
<style>
    .main { padding-top: 1.5rem; }
    .urdu-text { font-family: 'Noto Nastaliq Urdu', serif; direction: rtl; text-align: right; font-size: 20px; }
    .roman-text { font-family: 'Arial'; font-size: 18px; text-align: left; }
    .translation-output {
        background: #f7f7f9; padding: 15px; border-radius: 8px;
        border-left: 4px solid #1f77b4; margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# MODEL CLASSES (unchanged)
# --------------------------

class BPETokenizer:
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.idx_to_token = {}
        self.trained = False

    def get_word_tokens(self, text):
        words = text.strip().split()
        word_tokens = {}
        for word in words:
            chars = list(word) + ['</w>']
            token_word = ' '.join(chars)
            word_tokens[token_word] = word_tokens.get(token_word, 0) + 1
        return word_tokens

    def merge_tokens(self, pair, word_tokens):
        new_word_tokens = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in word_tokens:
            new_word = word.replace(bigram, replacement)
            new_word_tokens[new_word] = word_tokens[word]
        return new_word_tokens

    def encode(self, text):
        if not self.trained:
            return [0]
        word_tokens = self.get_word_tokens(text)
        for pair in self.merges:
            word_tokens = self.merge_tokens(pair, word_tokens)
        tokens = []
        for word in word_tokens:
            for token in word.split():
                tokens.append(self.vocab.get(token, self.vocab.get('<UNK>', 0)))
        return tokens

    def decode(self, indices):
        if not self.trained:
            return ""
        tokens = []
        for idx in indices:
            if idx in self.idx_to_token:
                tokens.append(self.idx_to_token[idx])
        text = ''.join(tokens)
        return text.replace('</w>', ' ').strip()


class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(BiLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, encoder_hidden_dim):
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_projection = nn.Linear(encoder_hidden_dim * 2, hidden_dim)
        self.cell_projection = nn.Linear(encoder_hidden_dim * 2, hidden_dim)

    def forward(self, x, hidden, cell):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.out(output)
        return output, (hidden, cell)

    def init_hidden(self, encoder_hidden, encoder_cell):
        batch_size = encoder_hidden.size(1)
        encoder_layers = encoder_hidden.size(0) // 2
        encoder_hidden = encoder_hidden.view(encoder_layers, 2, batch_size, -1)
        encoder_cell = encoder_cell.view(encoder_layers, 2, batch_size, -1)
        last_hidden = torch.cat([encoder_hidden[-1, 0], encoder_hidden[-1, 1]], dim=1)
        last_cell = torch.cat([encoder_cell[-1, 0], encoder_cell[-1, 1]], dim=1)
        hidden = self.hidden_projection(last_hidden).unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = self.cell_projection(last_cell).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return hidden, cell


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

# --------------------------
# LOAD MODEL SAFELY
# --------------------------

@st.cache_resource
def load_model_and_tokenizers():
    """Loads model and tokenizers safely"""
    try:
        model_path = "models/best_model.pth"
        urdu_tokenizer_path = "tokenizers/urdu_tokenizer.pkl"
        roman_tokenizer_path = "tokenizers/roman_tokenizer.pkl"

        if not all(os.path.exists(p) for p in [model_path, urdu_tokenizer_path, roman_tokenizer_path]):
            st.warning("⚠️ Model or tokenizer files not found. Please upload them to your repo.")
            return None, None, None

        checkpoint = torch.load(model_path, map_location='cpu')

        encoder_vocab_size = checkpoint['encoder.embedding.weight'].shape[0]
        decoder_vocab_size = checkpoint['decoder.embedding.weight'].shape[0]

        encoder = BiLSTMEncoder(encoder_vocab_size, 256, 512, 3, 0.3)
        decoder = LSTMDecoder(decoder_vocab_size, 256, 512, 3, 0.3, 512)
        model = Seq2SeqModel(encoder, decoder)
        model.load_state_dict(checkpoint)
        model.eval()

        with open(urdu_tokenizer_path, 'rb') as f:
            urdu_tokenizer = pickle.load(f)
        with open(roman_tokenizer_path, 'rb') as f:
            roman_tokenizer = pickle.load(f)

        st.success("✅ Model and tokenizers loaded successfully!")
        return model, urdu_tokenizer, roman_tokenizer

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None


# --------------------------
# TRANSLATION FUNCTION
# --------------------------

def translate_text(model, urdu_tokenizer, roman_tokenizer, text, max_length=100):
    if not model or not urdu_tokenizer or not roman_tokenizer:
        return "Model or tokenizers not loaded"

    try:
        input_tokens = urdu_tokenizer.encode(text)
        sos_token = roman_tokenizer.vocab.get('<SOS>', 1)
        eos_token = roman_tokenizer.vocab.get('<EOS>', 2)
        src = torch.tensor([input_tokens], dtype=torch.long)

        trg_indices = [sos_token]
        with torch.no_grad():
            encoder_outputs, (encoder_hidden, encoder_cell) = model.encoder(src)
            hidden, cell = model.decoder.init_hidden(encoder_hidden, encoder_cell)
            for _ in range(max_length):
                input_token = torch.tensor([[trg_indices[-1]]], dtype=torch.long)
                output, (hidden, cell) = model.decoder(input_token, hidden, cell)
                pred_token = output.argmax(2).item()
                trg_indices.append(pred_token)
                if pred_token == eos_token:
                    break

        return roman_tokenizer.decode(trg_indices)

    except Exception as e:
        return f"Translation error: {str(e)}"


# --------------------------
# STREAMLIT UI
# --------------------------

def main():
    st.title("🔤 Urdu → Roman Urdu Translator")
    st.markdown("**AI-powered translation from Urdu script to Roman Urdu**")

    with st.spinner("Loading model..."):
        model, urdu_tokenizer, roman_tokenizer = load_model_and_tokenizers()

    input_text = st.text_area("📝 Enter Urdu text here:", height=150, key="urdu_input")

    if st.button("🚀 Translate"):
        if not input_text.strip():
            st.warning("Please enter some Urdu text.")
        else:
            translation = translate_text(model, urdu_tokenizer, roman_tokenizer, input_text)
            st.markdown(f'<div class="translation-output roman-text">{translation}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
