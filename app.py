import streamlit as st
import pickle
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ✅ Minimal keras imports (no full tensorflow import)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- NLTK ----------
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = re.sub(r'[^a-z\\s]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# ---------- Load model & files ----------
model = load_model("stress_model_final_tfkeras.h5")

with open("tokenizer_tfkeras.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open("pipeline_config.pkl", "rb") as f:
    config = pickle.load(f)

MAX_LENGTH = config["max_length"]
THRESHOLD = config["threshold"]

def predict_stress(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    score = model.predict(pad, verbose=0)[0][0]

    if score > THRESHOLD:
        return "😟 STRESS DETECTED", score
    else:
        return "😊 NO STRESS", score

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Stress Detection App", page_icon="🧠")

st.title("🧠 Stress Detection using NLP")
st.write("Enter text to check stress level")

user_text = st.text_area("Enter your text")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:
        result, score = predict_stress(user_text)
        st.subheader(result)
        st.write(f"Confidence: **{score*100:.2f}%**")
