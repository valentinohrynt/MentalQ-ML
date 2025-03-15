from flask import Flask, request, jsonify
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import stanza
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models import Word2Vec
import logging
import warnings
import os
import re
import string

# ============ GPU CONFIGURATION ============ #
# Check and enable GPU for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Prevents memory overflow
        print("✅ GPU is available and configured")
    except RuntimeError as e:
        print(f"⚠️ GPU Error: {e}")

# Enable XLA (Accelerated Linear Algebra) for optimization
tf.config.optimizer.set_jit(True)

# ============ LOGGING & WARNINGS ============ #
logging.basicConfig(level=logging.WARNING)
logging.getLogger('stanza').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============ STANZA INITIALIZATION ============ #
# Load Stanza with GPU support
nlp = stanza.Pipeline('id', processors='tokenize,lemma', use_gpu=True)

# ============ LOAD MODELS & ENCODERS ============ #
# Load trained LSTM model
model = load_model('model_save_ml/ml_model_lstm.h5')

# Load Label Encoder
with open('data/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load Word2Vec model
word2vec_model = Word2Vec.load("model_word2vec/word2vec_model_MentalQ.model")
word_index = {word: i + 1 for i, word in enumerate(word2vec_model.wv.index_to_key)}
embedding_dim = 100
max_sequence_length = 100

# ============ TEXT PREPROCESSING FUNCTIONS ============ #
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside brackets
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]\(.*?\)', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation again
    text = re.sub(r'\n', ' ', text)  # Remove new lines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

def remove_stopwords(text):
    factory = StopWordRemoverFactory()
    stop_words = set(factory.get_stop_words())

    manual_stopwords = {"aku", "kamu", "dia", "mereka", "kita", "kami", "mu", "ku", "nya", "itu", "ini", "sini", "situ", "sana", "begitu", "yaitu", "yakni"}
    stop_words.update(manual_stopwords)
    
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

def lemmatize_and_tokenize_text(text):
    doc = nlp(text)
    tokens = []
    lemmatized_text = []
    for sentence in doc.sentences:
        for word in sentence.words:
            tokens.append(word.text)  
            lemmatized_text.append(word.lemma)  
    return lemmatized_text, tokens

def text_to_sequence(tokens, word_index):
    return [word_index[word] for word in tokens if word in word_index]

def preprocess_input(text_raw):
    text_raw = clean_text(text_raw)
    text_raw = text_raw.lower()
    text_raw = remove_stopwords(text_raw)
    lemmatized_text, tokenized_text = lemmatize_and_tokenize_text(text_raw)
    sequence = text_to_sequence(tokenized_text, word_index)
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='post')
    return padded_sequence

# ============ PREDICTION FUNCTION ============ #
def predict_status_with_probabilities(text_raw):
    preprocessed_input = preprocess_input(text_raw)
    predicted_class_probs = model.predict(preprocessed_input)
    predicted_class_idx = np.argmax(predicted_class_probs, axis=1)
    predicted_class_prob = np.max(predicted_class_probs, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class_idx)
    return predicted_label[0], predicted_class_probs[0]

# ============ FLASK API ============ #
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid input, expected JSON format"}), 400
    
    try:
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

# ============ START SERVER ============ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
