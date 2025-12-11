
---

# `app.py`
```python
import os
import re
import pickle
import streamlit as st

# Optional: import torch only if available / needed
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ------------------------------
# Configuration
# ------------------------------

URDU_TOKENIZER_FILE =( "experiment_1_urdu_tokenizer.pkl")
ROMAN_TOKENIZER_FILE = ( "experiment_1_roman_tokenizer.pkl")
MODEL_FILE =( "best_model.pth")  # optional

# ------------------------------
# Rule-based transliteration map
# (simple, readable Roman Urdu)
# ------------------------------
URDU_ROMAN_MAP = {
    'ا': 'a', 'آ': 'aa', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's',
    'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd', 'ذ': 'z',
    'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's',
    'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
    'ک': 'k', 'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'o',
    'ؤ': 'o', 'ہ': 'h', 'ھ': 'h', 'ء': "'", 'ی': 'y', 'ے': 'e',
    '۔': '.', '،': ',', '؟': '?', '؛': ';', '\n': ' ', '\t': ' '
}

def rule_based_roman(text: str) -> str:
    out_chars = []
    for ch in text:
        out_chars.append(URDU_ROMAN_MAP.get(ch, ch))
    out = ''.join(out_chars)
    # Basic cleanup & common replacements for readability
    out = re.sub(r'\s+', ' ', out).strip()
    # normalize some vowel repeats
    out = out.replace('aa', 'a')  # makes ā -> a (simple heuristic)
    out = out.replace(" '", "'")
    return out

# ------------------------------
# Attempt to load tokenizers & model (if present)
# ------------------------------
loaded_tokenizers = False
urdu_tokenizer = None
roman_tokenizer = None
model = None

def try_load_pickles():
    global urdu_tokenizer, roman_tokenizer, loaded_tokenizers
    try:
        if os.path.exists(URDU_TOKENIZER_FILE):
            with open(URDU_TOKENIZER_FILE, "rb") as f:
                urdu_tokenizer = pickle.load(f)
        if os.path.exists(ROMAN_TOKENIZER_FILE):
            with open(ROMAN_TOKENIZER_FILE, "rb") as f:
                roman_tokenizer = pickle.load(f)
        loaded_tokenizers = (urdu_tokenizer is not None and roman_tokenizer is not None)
    except Exception as e:
        loaded_tokenizers = False
        st.warning(f"Could not load tokenizer pickles: {e}")

def try_load_model():
    global model
    if not TORCH_AVAILABLE:
        return False
    try:
        if os.path.exists(MODEL_FILE):
            model = torch.load(MODEL_FILE, map_location="cpu")
            # If you have a custom model class, you must load state_dict into it here.
            return True
    except Exception as e:
        st.warning(f"Model load failed: {e}")
    return False

try_load_pickles()
model_loaded = try_load_model()

# ------------------------------
# Inference wrapper (placeholder)
# ------------------------------
def neural_infer(text: str) -> str:
    """
    Placeholder for neural model inference. The concrete implementation depends on:
    - your model architecture/class
    - tokenizers' encode/decode interfaces
    Replace this function with proper inference if you have a trained model.
    """
    # This is a fallback: if you have tokenizers & a model, you should implement
    # encoding, model forward, and decoding/greedy-beam search here.
    # For now, we simply return rule-based output with a note.
    return rule_based_roman(text)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Urdu → Roman Urdu", layout="wide")
st.title("Urdu → Roman Urdu Translator")
st.markdown(
    "Type or paste Urdu text (left) and get Roman Urdu (right). "
    "If you have model/tokenizer files in `models/`, the app will try to use them."
)

col1, col2 = st.columns([1, 1])

with col1:
    urdu_input = st.text_area("Enter Urdu text", height=300, placeholder="یہاں اردو لکھیں...")
    use_neural = st.checkbox("Use neural model (if available)", value=False)
    if st.button("Convert"):
        if not urdu_input.strip():
            st.warning("Please enter some Urdu text.")
        else:
            if use_neural and model_loaded and loaded_tokenizers:
                roman_output = neural_infer(urdu_input)
            else:
                roman_output = rule_based_roman(urdu_input)

            st.success("Converted ✅")
            # show below

with col2:
    st.text_area("Roman Urdu output", value=(roman_output if 'roman_output' in locals() else ""), height=300)

# ------------------------------
# Utilities: Save output or sample
# ------------------------------
st.write("---")
st.subheader("Utilities")
if st.button("Save last output to file"):
    if 'roman_output' not in locals():
        st.warning("No conversion yet. Press Convert first.")
    else:
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)
        idx = len(os.listdir(out_dir)) + 1
        out_path = os.path.join(out_dir, f"roman_{idx}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(urdu_input + "\n\n" + roman_output)
        st.success(f"Saved to `{out_path}`")

st.markdown("**Model status**")
st.write(f"- Tokenizers loaded: `{loaded_tokenizers}`")
st.write(f"- PyTorch available: `{TORCH_AVAILABLE}`")
st.write(f"- Model loaded from `{MODEL_FILE}`: `{model_loaded}`")

st.info("If you want full neural inference, modify `neural_infer()` to match your model class and decoding logic (tokenizer.encode -> model -> tokenizer.decode).")
