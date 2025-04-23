
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


try:
    with open('model/spam_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except (FileNotFoundError, EOFError):
    model = None
    vectorizer = None

def train_model():
    """Train a simple spam classification model if none exists"""
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    
    # Example dataset (in a real app, use a larger dataset)
    texts = [
        "Free money now!!!", "Hi, how are you?", "Win a free iPhone!", 
        "Meeting at 3pm tomorrow", "Claim your prize today", 
        "Your account has been compromised", "Lunch tomorrow?", 
        "URGENT: Click this link", "Project update", "Limited time offer"
    ]
    labels = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 1=spam, 0=ham
    
    # Vectorize the text
    global vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    # Train the model
    global model
    model = MultinomialNB()
    model.fit(X, labels)
    
    # Save the model and vectorizer
    with open('model/spam_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

# Train model if not loaded
if model is None or vectorizer is None:
    train_model()

def check_spam(text):
    
    
    # Vectorize the input text
    text_vector = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vector)
    probability = model.predict_proba(text_vector)
    
    # Get confidence percentage
    confidence = round(np.max(probability) * 100, 2)
    
    result = {
        'is_spam': bool(prediction[0]),
        'confidence': confidence,
        'text': text
    }
    
    return result



