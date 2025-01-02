import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Load models and vocab info
dl_model = load_model('models/dl_model.h5')  # Deep learning model
rfc = pickle.load(open('models/rfc_model.pkl', 'rb'))  # Machine learning model
tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))  # TF-IDF vectorizer
le = pickle.load(open('models/label.pkl', 'rb'))  # Label encoder for inverse transformation
vocab_info = pickle.load(open('models/vocab_info.pkl', 'rb'))  # Vocab info for deep learning model

# Function for cleaning text
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# Deep learning model text cleaning function
def sentence_cleaning(sentence):
    corpus = []
    text = re.sub("[^a-zA-Z]", " ", sentence)
    text = text.lower()
    text = text.split()
    text = [PorterStemmer().stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    corpus.append(text)
    one_hot_word = [one_hot(input_text=word, n=vocab_info['vocab_size']) for word in corpus]  # Use vocab size from loaded vocab_info
    pad = pad_sequences(sequences=one_hot_word, maxlen=vocab_info['max_len'], padding='pre')  # Use max_len from vocab_info
    return pad

# Function for deep learning prediction
def predict_dl_emotion(input_text):
    sentence = sentence_cleaning(input_text)
    result = le.inverse_transform(np.argmax(dl_model.predict(sentence), axis=-1))[0]
    proba = np.max(dl_model.predict(sentence))
    return result, proba

# Function for machine learning prediction
def predict_ml_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = rfc.predict(input_vectorized)[0]
    predicted_emotion = le.inverse_transform([predicted_label])[0]
    proba = np.max(rfc.predict_proba(input_vectorized))
    return predicted_emotion, proba

# Streamlit app
st.title("Six Human Emotions Classification App - ML & DL Models")

# Model selection
model_choice = st.selectbox("Select Model", ("Deep Learning", "Machine Learning"))

# Input text from user
user_input = st.text_input("Enter a sentence:")

if st.button("Predict Emotion"):
    if model_choice == "Deep Learning":
        pred_emotion, proba = predict_dl_emotion(user_input)
        st.write(f"Predicted Emotion: {pred_emotion}")
        st.write(f"Probability: {proba:.4f}")
    else:
        pred_emotion, proba = predict_ml_emotion(user_input)
        st.write(f"Predicted Emotion: {pred_emotion}")
        st.write(f"Probability: {proba:.4f}")
