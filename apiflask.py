from flask import Flask, request, jsonify
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.tokenize import word_tokenize
import re
import string
import stanza
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models import Word2Vec
import logging
import warnings

# Silence warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger('stanza').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Inisialisasi
nlp = stanza.Pipeline('id', processors='tokenize,lemma', use_gpu=False)

# Load model yang sudah dilatih
model = load_model('model_save_ml/ml_model_lstm.h5')

# Load Label Encoder
with open('data/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load Word2Vec model dan word_index
word2vec_model = Word2Vec.load("model_word2vec/word2vec_model_MentalQ.model")
word_index = {word: i + 1 for i, word in enumerate(word2vec_model.wv.index_to_key)}
embedding_dim = 100
max_sequence_length = 100

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Fungsi untuk menghapus stopwords
def remove_stopwords(text):
    factory = StopWordRemoverFactory()
    stop_words = set(factory.get_stop_words())

    manual_stopwords = {"aku", "kamu", "dia", "mereka", "kita", "kami", "mu", "ku", "nya", "itu", "ini", "sini", "situ", "sana", "begitu", "yaitu", "yakni"}
    stop_words.update(manual_stopwords)
    
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Fungsi untuk lemmatization
def lemmatize_text(text):
    doc = nlp(text)
    sentences = []
    for sentence in doc.sentences:
        lemmas = [word.lemma for word in sentence.words]
        sentences.append(' '.join(lemmas))
    return ' '.join(sentences)

# Fungsi untuk mengonversi token menjadi sequence of integers
def text_to_sequence(tokens, word_index):
    return [word_index[word] for word in tokens if word in word_index]

# Fungsi untuk preprocessing teks
def preprocess_input(text_raw):
    # Cleaning text
    text_raw = clean_text(text_raw)
    
    # Convert to lowercase
    text_raw = text_raw.lower()
    
    # Remove stopwords
    text_raw = remove_stopwords(text_raw)
    
    # Lemmatize text
    text_preprocessed = lemmatize_text(text_raw)
    
    # Tokenized text
    tokenized_text = word_tokenize(text_preprocessed)
    
    # Convert tokens to sequence of integers
    sequence = text_to_sequence(tokenized_text, word_index)
    
    # Padding sequence
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='post')
    
    return padded_sequence

# Fungsi untuk melakukan prediksi dengan probabilitas
def predict_status_with_probabilities(text_raw):
    # Preprocess input text
    preprocessed_input = preprocess_input(text_raw)
    
    # Prediksi kelas dengan probabilitas
    predicted_class_probs = model.predict(preprocessed_input)
    
    # Mendapatkan kelas terprediksi dengan probabilitas tertinggi
    predicted_class_idx = np.argmax(predicted_class_probs, axis=1)
    
    # Mendapatkan probabilitas untuk kelas terprediksi
    predicted_class_prob = np.max(predicted_class_probs, axis=1)
    
    # Mengembalikan label kelas yang terprediksi dan probabilitas untuk setiap kelas
    predicted_label = label_encoder.inverse_transform(predicted_class_idx)
    
    return predicted_label[0], predicted_class_probs[0]

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Ensure request contains JSON
    if not request.is_json:
        return jsonify({"error": "Invalid input, expected JSON format"}), 400
    
    try:
        # Get input data
        data = request.get_json()
        
        if 'statements' not in data:
            return jsonify({"error": "Missing 'statements' in request"}), 400
        
        statements = data['statements']
        
        if not isinstance(statements, list):
            return jsonify({"error": "Input data must be a list of statements"}), 400
        
        response = []

        for statement in statements:
            predicted_status, class_probabilities = predict_status_with_probabilities(statement)
            confidence_scores = {label: float(prob) for label, prob in zip(label_encoder.classes_, class_probabilities)}
            
            response.append({
                "confidence_scores": confidence_scores,
                "predicted_status": predicted_status,
                "statement": statement
            })
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
