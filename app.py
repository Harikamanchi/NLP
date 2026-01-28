import streamlit as st
import pickle
import re
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load model
model = load_model("stress_bilstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_basic(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop_words])

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()])

def preprocess(text):
    text = clean_basic(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

st.title("🧠 Stress Detection using NLP")
st.write("Enter a sentence to detect stress level")

user_input = st.text_area("Enter text here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        processed = preprocess(user_input)
        seq = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=100)
        pred = model.predict(padded)[0][0]

        if pred > 0.5:
            st.error("⚠️ Stress Detected")
        else:
            st.success("✅ No Stress Detected")
