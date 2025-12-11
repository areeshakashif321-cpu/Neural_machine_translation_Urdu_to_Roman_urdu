import os, re, pickle, streamlit as st
try:
    import torch
    TORCH = True
except:
    TORCH = False

FILES = [
    "best_model.pth","experiment_1_model.pth","experiment_2_model.pth",
    "experiment_1_urdu_tokenizer.pkl","experiment_1_roman_tokenizer.pkl",
    "experiment_2_urdu_tokenizer.pkl","experiment_2_roman_tokenizer.pkl"
]

ROOT = os.getcwd()
DIRS = [ROOT, os.path.join(ROOT, "models")]
found = {}
for d in DIRS:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f in FILES:
                found[f] = os.path.join(d, f)

st.set_page_config(page_title="Urdu → Roman Urdu", layout="wide")
st.title("Urdu → Roman Urdu Translator")

UR_MAP = {
 'ا':'a','آ':'aa','ب':'b','پ':'p','ت':'t','ٹ':'t','ث':'s','ج':'j','چ':'ch',
 'ح':'h','خ':'kh','د':'d','ڈ':'d','ذ':'z','ر':'r','ڑ':'r','ز':'z','ژ':'zh',
 'س':'s','ش':'sh','ص':'s','ض':'z','ط':'t','ظ':'z','ع':'a','غ':'gh','ف':'f',
 'ق':'q','ک':'k','گ':'g','ل':'l','م':'m','ن':'n','ں':'n','و':'o','ؤ':'o',
 'ہ':'h','ھ':'h','ء':"'",'ی':'y','ے':'e','۔':'.','،':',','؟':'?'
}

def rule_roman(t):
    o = ''.join(UR_MAP.get(ch, ch) for ch in t)
    o = re.sub(r'\s+', ' ', o)
    return o.strip()

def load_tok(p):
    try:
        return pickle.load(open(p, "rb"))
    except:
        return None

u_tok = None
r_tok = None
for f in found:
    if f.endswith("urdu_tokenizer.pkl"): u_tok = load_tok(found[f])
    if f.endswith("roman_tokenizer.pkl"): r_tok = load_tok(found[f])

model = None
mp = None
for f in ["best_model.pth","experiment_1_model.pth","experiment_2_model.pth"]:
    if f in found:
        mp = found[f]
        break
if mp and TORCH:
    try:
        model = torch.load(mp, map_location="cpu")
        model.eval()
    except:
        model = None

def neural(t):
    return rule_roman(t)

c1, c2 = st.columns(2)
text = c1.text_area("Urdu", height=300)
use = c1.checkbox("Use neural", value=False)
btn = c1.button("Convert")
box = c2.empty()

if btn:
    if use and model and u_tok and r_tok:
        out = neural(text)
    else:
        out = rule_roman(text)
    box.text_area("Roman Urdu", out, height=300)

