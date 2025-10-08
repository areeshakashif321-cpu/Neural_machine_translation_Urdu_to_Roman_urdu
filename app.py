# Last updated: 2025-09-21 15:47
import streamlit as st
import torch
import torch.nn as nn
import pickle
import json
import pandas as pd
from collections import defaultdict
import re
import os

# Set page config
st.set_page_config(
    page_title="Urdu to Roman Urdu Translator",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTextArea textarea {
        font-size: 18px !important;
        font-family: 'Noto Nastaliq Urdu', serif !important;
    }
    .translation-output {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
        font-size: 18px;
        font-family: 'Arial', sans-serif;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .experiment-results {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 20px;
        direction: rtl;
        text-align: right;
    }
    .roman-text {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

class BPETokenizer:
    """Byte Pair Encoding Tokenizer"""
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
            return [0]  # Return padding token if not trained

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
            if idx in self.idx_to_token and idx not in [
                self.vocab.get('<PAD>', 0), self.vocab.get('<SOS>', 1),
                self.vocab.get('<EOS>', 2), self.vocab.get('<UNK>', 3)
            ]:
                tokens.append(self.idx_to_token[idx])

        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM Encoder"""
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
    """LSTM Decoder with attention"""
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
    """Sequence to Sequence Model"""
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)

        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(src)
        hidden, cell = self.decoder.init_hidden(encoder_hidden, encoder_cell)

        input_token = trg[:, 0:1]

        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(input_token, hidden, cell)
            outputs[:, t:t+1] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input_token = trg[:, t:t+1] if teacher_force else top1

        return outputs

@st.cache_resource
def load_model_and_tokenizers():
    """Load trained model and tokenizers with error handling"""
    try:
        # Updated paths based on Colab structure
        base_dir = '/content/UrduToRomanUrduTranslator/UrduToRomanUrduTranslator'
        model_paths = [
            f'{base_dir}/best_model.pth',
            f'{base_dir}/experiment_1_model.pth',
            f'{base_dir}/experiment_2_model.pth',
            f'{base_dir}/experiment_3_model.pth',
            '/content/best_model.pth',
            'models/best_model.pth',
            './models/best_model.pth'
        ]

        tokenizer_paths = [
            (f'{base_dir}/experiment_1_urdu_tokenizer.pkl', f'{base_dir}/experiment_1_roman_tokenizer.pkl'),
            (f'{base_dir}/experiment_2_urdu_tokenizer.pkl', f'{base_dir}/experiment_2_roman_tokenizer.pkl'),
            (f'{base_dir}/experiment_3_urdu_tokenizer.pkl', f'{base_dir}/experiment_3_roman_tokenizer.pkl'),
            ('urdu_tokenizer.pkl', 'roman_tokenizer.pkl'),
            ('./urdu_tokenizer.pkl', './roman_tokenizer.pkl'),
            ('tokenizers/urdu_tokenizer.pkl', 'tokenizers/roman_tokenizer.pkl'),
            ('./tokenizers/urdu_tokenizer.pkl', './tokenizers/roman_tokenizer.pkl')
        ]

        model = None
        urdu_tokenizer = None
        roman_tokenizer = None

        # Try to load model
        for path in model_paths:
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location='cpu')

                    # Your model was saved as state_dict directly
                    # Need to determine vocab sizes from the embedding weights
                    encoder_vocab_size = checkpoint['encoder.embedding.weight'].shape[0]
                    decoder_vocab_size = checkpoint['decoder.embedding.weight'].shape[0]

                    # Use 3 layers as per experiment configs (encoder_layers=3 in exp3, but to match saved)
                    # Assume 3 layers for compatibility with saved models
                    encoder = BiLSTMEncoder(
                        vocab_size=encoder_vocab_size,
                        embed_dim=256,
                        hidden_dim=512,
                        num_layers=3,  # Adjusted to 3 for compatibility
                        dropout=0.3
                    )

                    decoder = LSTMDecoder(
                        vocab_size=decoder_vocab_size,
                        embed_dim=256,
                        hidden_dim=512,
                        num_layers=3,  # Adjusted to 3 for compatibility
                        dropout=0.3,
                        encoder_hidden_dim=512
                    )

                    model = Seq2SeqModel(encoder, decoder)

                    # Load the state dict directly (checkpoint IS the state dict)
                    model.load_state_dict(checkpoint)
                    model.eval()

                    st.success(f"✅ Model loaded successfully from {path}")
                    st.info(f"Encoder vocab size: {encoder_vocab_size}, Decoder vocab size: {decoder_vocab_size}")
                    break

                except Exception as e:
                    st.warning(f"Failed to load model from {path}: {e}")
                    continue

        # Try to load tokenizers
        for urdu_path, roman_path in tokenizer_paths:
            if os.path.exists(urdu_path) and os.path.exists(roman_path):
                try:
                    with open(urdu_path, 'rb') as f:
                        urdu_tokenizer = pickle.load(f)
                    with open(roman_path, 'rb') as f:
                        roman_tokenizer = pickle.load(f)

                    st.success(f"✅ Tokenizers loaded from {urdu_path} and {roman_path}")
                    break
                except Exception as e:
                    st.warning(f"Failed to load tokenizers from {urdu_path}, {roman_path}: {e}")
                    continue

        return model, urdu_tokenizer, roman_tokenizer

    except Exception as e:
        st.error(f"Error loading model and tokenizers: {e}")
        return None, None, None

def translate_text(model, urdu_tokenizer, roman_tokenizer, text, max_length=100):
    """Translate Urdu text to Roman Urdu"""
    if not model or not urdu_tokenizer or not roman_tokenizer:
        return "Model or tokenizers not loaded"

    try:
        # Tokenize input
        input_tokens = urdu_tokenizer.encode(text)

        # Add SOS and EOS tokens
        sos_token = roman_tokenizer.vocab.get('<SOS>', 1)
        eos_token = roman_tokenizer.vocab.get('<EOS>', 2)

        # Convert to tensor
        src = torch.tensor([input_tokens], dtype=torch.long)

        # Initialize decoder input with SOS token
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

        # Decode the output
        roman_text = roman_tokenizer.decode(trg_indices)
        return roman_text

    except Exception as e:
        return f"Translation error: {str(e)}"

def create_demo_examples():
    """Create demo examples for testing"""
    return [
        "آپ کا نام کیا ہے؟",
        "میں اردو سیکھ رہا ہوں",
        "آج موسم بہت اچھا ہے",
        "کتاب پڑھنا مفید ہے",
        "خوش آمدید"
    ]

def main():
    """Main Streamlit app"""
    # Title and description
    st.title("🔤 Urdu to Roman Urdu Translator")
    st.markdown("**AI-powered translation from Urdu script to Roman Urdu**")

    # Load model and tokenizers
    with st.spinner("Loading AI model and tokenizers..."):
        model, urdu_tokenizer, roman_tokenizer = load_model_and_tokenizers()

    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This app translates Urdu text written in Arabic script to Roman Urdu
        (Urdu written using Latin alphabet).

        **Features:**
        - Neural sequence-to-sequence model
        - BPE tokenization
        - BiLSTM encoder-decoder architecture
        """)

        st.header("🎯 Model Status")
        if model and urdu_tokenizer and roman_tokenizer:
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model not loaded")
            st.warning("Please ensure model files are in the correct directory")

        st.header("📊 Model Info")
        if model:
            st.info(f"Encoder: BiLSTM\nDecoder: LSTM\nDevice: CPU")

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Input (Urdu)")

        # Demo examples
        st.subheader("Quick Examples:")
        demo_examples = create_demo_examples()

        example_buttons = []
        for i, example in enumerate(demo_examples):
            if st.button(f"{example}", key=f"example_{i}"):
                st.session_state.input_text = example

        # Text input
        input_text = st.text_area(
            "Enter Urdu text:",
            value=st.session_state.get('input_text', ''),
            height=150,
            key="urdu_input",
            help="Type or paste Urdu text in Arabic script"
        )

        # Display input in styled format
        if input_text:
            st.markdown(
                f'<div class="urdu-text">{input_text}</div>',
                unsafe_allow_html=True
            )

    with col2:
        st.header("🔄 Output (Roman Urdu)")

        if input_text and st.button("🚀 Translate", type="primary"):
            with st.spinner("Translating..."):
                if model and urdu_tokenizer and roman_tokenizer:
                    translation = translate_text(
                        model, urdu_tokenizer, roman_tokenizer, input_text
                    )

                    # Display translation
                    st.markdown(
                        f'<div class="translation-output roman-text">{translation}</div>',
                        unsafe_allow_html=True
                    )

                    # Store in session state for persistence
                    st.session_state.last_translation = translation

                else:
                    st.error("❌ Model not available. Please check model files.")

        # Display last translation if available
        elif hasattr(st.session_state, 'last_translation'):
            st.markdown(
                f'<div class="translation-output roman-text">{st.session_state.last_translation}</div>',
                unsafe_allow_html=True
            )

    # Additional features
    st.header("🔧 Additional Features")

    tab1, tab2, tab3 = st.tabs(["Batch Translation", "Model Details", "Help"])

    with tab1:
        st.subheader("Batch Translation")

        batch_text = st.text_area(
            "Enter multiple lines of Urdu text (one per line):",
            height=100,
            help="Each line will be translated separately"
        )

        if batch_text and st.button("Translate All"):
            lines = batch_text.strip().split('\n')
            results = []

            progress_bar = st.progress(0)
            for i, line in enumerate(lines):
                if line.strip():
                    if model and urdu_tokenizer and roman_tokenizer:
                        translation = translate_text(
                            model, urdu_tokenizer, roman_tokenizer, line.strip()
                        )
                        results.append({'Urdu': line.strip(), 'Roman Urdu': translation})
                    progress_bar.progress((i + 1) / len(lines))

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="translations.csv",
                    mime="text/csv"
                )

    with tab2:
        st.subheader("Model Architecture Details")

        if model:
            st.code("""
            Model Architecture:
            ==================
            Encoder: Bidirectional LSTM
            - Embedding Dimension: 256
            - Hidden Dimension: 512
            - Number of Layers: 3
            - Dropout: 0.3

            Decoder: LSTM with Attention
            - Embedding Dimension: 256
            - Hidden Dimension: 512
            - Number of Layers: 3
            - Dropout: 0.3

            Tokenizer: Byte Pair Encoding (BPE)
            - Vocabulary Size: ~2000 tokens
            """)
        else:
            st.warning("Model details not available - model not loaded")

    with tab3:
        st.subheader("Help & Instructions")

        st.markdown("""
        ### How to use this translator:

        1. **Input**: Type or paste Urdu text in Arabic script in the input box
        2. **Translate**: Click the "🚀 Translate" button
        3. **Output**: View the Roman Urdu translation in the output box

        ### Tips for better translations:
        - Keep sentences reasonably short
        - Use proper Urdu spelling and grammar
        - The model works best with common words and phrases

        ### Troubleshooting:
        - If you see "Model not loaded", check that model files are in the correct directory
        - For empty translations, try rephrasing your input
        - Contact support if issues persist

        ### File Requirements:
        The following files should be in your app directory:
        - `best_model.pth` - Trained model weights
        - `urdu_tokenizer.pkl` - Urdu tokenizer
        - `roman_tokenizer.pkl` - Roman Urdu tokenizer
        """)

if __name__ == "__main__":
    main()
