import streamlit as st
import pickle
import re
import nltk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# NLTK setup
# =========================
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# =========================
# Load model & tokenizer
# =========================
model = load_model("stress_bilstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# Text preprocessing
# (MUST match training)
# =========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Stress Detection using NLP", layout="centered")

st.title("🧠 Stress Detection using NLP")
st.write("Enter a sentence to detect stress level")

user_input = st.text_area(
    "Enter text here",
    placeholder="Type a sentence like: I am feeling calm and relaxed today..."
)

# Threshold (tuned to reduce false stress alerts)
THRESHOLD = 0.6

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        # Preprocess
        processed = preprocess_text(user_input)

        # Tokenize & pad
        seq = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=100)

        # Predict probability
        pred_prob = model.predict(padded)[0][0]

        # Show probability
        st.write(f"🔍 **Stress Probability:** `{pred_prob:.2f}`")

        # Decision
        if pred_prob >= THRESHOLD:
            st.error("⚠️ Stress Detected")
        else:
            st.success("✅ No Stress Detected")

        # Optional explanation
        st.caption(
            "ℹ️ Prediction is probability-based. Borderline cases may vary due to model limitations."
        )

