# app.py
import os
import re
import pickle
import streamlit as st

# Try import torch (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ------------------------------
# Files expected (single-file mode)
# ------------------------------
FILE_CANDIDATES = [
    "best_model.pth",
    "experiment_1_model.pth",
    "experiment_1_results.json",
    "experiment_1_roman_tokenizer.pkl",
    "experiment_1_urdu_tokenizer.pkl",
    "experiment_2_model.pth",
    "experiment_2_results.json",
    "experiment_2_roman_tokenizer.pkl",
    "experiment_2_urdu_tokenizer.pkl"
]

# ------------------------------
# Detect files in root or models/
# ------------------------------
ROOT = os.getcwd()
MODEL_DIRS_TO_CHECK = [ROOT, os.path.join(ROOT, "models")]

found_files = {}
for d in MODEL_DIRS_TO_CHECK:
    if not os.path.exists(d):
        continue
    for fname in os.listdir(d):
        if fname in FILE_CANDIDATES:
            found_files[fname] = os.path.join(d, fname)

st.set_page_config(page_title="Urdu → Roman Urdu", layout="wide")
st.title("Urdu → Roman Urdu Translator")

st.markdown(
    "Place your single files (e.g. `best_model.pth`, `experiment_1_urdu_tokenizer.pkl`, "
    "`experiment_1_roman_tokenizer.pkl`) in the repository root (or `models/`). "
    "The app will auto-detect and attempt to use them. If missing, a rule-based fallback is used."
)

# Show detected files
if found_files:
    st.write("Detected files:")
    for k, v in found_files.items():
        st.write(f"- `{k}`  →  `{v}`")
else:
    st.info("No model/tokenizer files detected. The app will use rule-based transliteration.")

# ------------------------------
# Rule-based transliteration map
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
    out = re.sub(r'\s+', ' ', out).strip()
    # Heuristics to make output more readable
    out = out.replace(' aa', ' a').replace('aa', 'a')
    out = out.replace('  ', ' ')
    return out

# ------------------------------
# Try load tokenizers (pickles) from found_files
# ------------------------------
urdu_tokenizer = None
roman_tokenizer = None
loaded_tokenizers = False

def try_load_tokenizer(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Failed to load tokenizer `{path}`: {e}")
        return None

# prefer explicit files if present, else try any *_urdu_tokenizer.pkl pattern in root
for fname in ["experiment_1_urdu_tokenizer.pkl", "experiment_2_urdu_tokenizer.pkl"]:
    if fname in found_files:
        urdu_tokenizer = try_load_tokenizer(found_files[fname])
        break

for fname in ["experiment_1_roman_tokenizer.pkl", "experiment_2_roman_tokenizer.pkl"]:
    if fname in found_files:
        roman_tokenizer = try_load_tokenizer(found_files[fname])
        break

# fallback: load any .pkl in root that might be tokenizer
if not (urdu_tokenizer and roman_tokenizer):
    for d in MODEL_DIRS_TO_CHECK:
        try:
            for fname in os.listdir(d):
                if fname.endswith("_urdu_tokenizer.pkl") and not urdu_tokenizer:
                    urdu_tokenizer = try_load_tokenizer(os.path.join(d, fname))
                if fname.endswith("_roman_tokenizer.pkl") and not roman_tokenizer:
                    roman_tokenizer = try_load_tokenizer(os.path.join(d, fname))
        except FileNotFoundError:
            pass

loaded_tokenizers = (urdu_tokenizer is not None and roman_tokenizer is not None)

# ------------------------------
# Try load model (any .pth found)
# ------------------------------
model = None
model_loaded = False
model_path = None

# prefer best_model.pth or experiment_1_model.pth if present
for candidate in ["best_model.pth", "experiment_1_model.pth", "experiment_2_model.pth"]:
    if candidate in found_files:
        model_path = found_files[candidate]
        break

# fallback: any .pth in directories
if model_path is None:
    for d in MODEL_DIRS_TO_CHECK:
        try:
            for fname in os.listdir(d):
                if fname.endswith(".pth"):
                    model_path = os.path.join(d, fname)
                    break
        except FileNotFoundError:
            pass
        if model_path:
            break

if model_path and TORCH_AVAILABLE:
    try:
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        model_loaded = True
        st.success(f"Model loaded from `{model_path}`")
    except Exception as e:
        st.warning(f"Could not load model `{model_path}`: {e}")
        model_loaded = False
else:
    if model_path:
        st.warning(f"Found model file `{model_path}` but PyTorch not available; install torch to use it.")
    else:
        st.info("No model (.pth) file detected.")

st.write("---")
st.markdown("### Conversion")

col1, col2 = st.columns(2)
with col1:
    urdu_input = st.text_area("Enter Urdu text", height=300, placeholder="یہاں اردو لکھیں...")
    use_neural = st.checkbox("Use neural model (if available)", value=False)
    convert_btn = st.button("Convert")

with col2:
    roman_output_box = st.empty()

def neural_infer_placeholder(text: str) -> str:
    """
    Placeholder - implement your specific model inference here if your model class is known.
    Currently returns rule-based transliteration.
    """
    # If you have loaded tokenizers and a model, integrate encode->model->decode here.
    # For example:
    # tokens = urdu_tokenizer.encode(text)
    # preds = model.generate(tokens)  # pseudo
    # out = roman_tokenizer.decode(preds)
    # return out
    return rule_based_roman(text)

if convert_btn:
    if not urdu_input.strip():
        st.warning("Please enter some Urdu text.")
    else:
        if use_neural and model_loaded and loaded_tokenizers:
            output_text = neural_infer_placeholder(urdu_input)
        else:
            output_text = rule_based_roman(urdu_input)
        roman_output_box.text_area("Roman Urdu output", value=output_text, height=300)

        # show save option
        if st.button("Save output to outputs/roman.txt"):
            os.makedirs("outputs", exist_ok=True)
            out_path = os.path.join("outputs", "roman.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("Urdu:\n")
                f.write(urdu_input + "\n\n")
                f.write("Roman:\n")
                f.write(output_text + "\n")
            st.success(f"Saved to `{out_path}`")

st.write("---")
st.subheader("Detected file status")
st.write(f"- Tokenizers loaded: `{loaded_tokenizers}`")
st.write(f"- Model loaded (torch): `{model_loaded}`")
if model_path:
    st.write(f"- Model path: `{model_path}`")
st.write("Place single files directly in the repository root (or `models/`) with names like `experiment_1_urdu_tokenizer.pkl`, `experiment_1_roman_tokenizer.pkl`, `best_model.pth`.")
