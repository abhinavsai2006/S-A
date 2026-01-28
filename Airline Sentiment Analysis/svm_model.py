# Airline Sentiment Analysis - SVM Model
# This script trains and tests a Support Vector Machine model for airline sentiment classification.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the airline sentiment dataset."""
    # Load the dataset
    df = pd.read_csv('Tweets.csv')

    # Map sentiment labels to numerical values
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_label'] = df['airline_sentiment'].map(sentiment_mapping)

    # Separate features and target
    X = df['text']
    y = df['sentiment_label']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def train_and_evaluate_model():
    """Train and evaluate the SVM model."""
    print("Loading and preprocessing airline sentiment dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Text vectorization using TF-IDF
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training SVM model...")
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print(f"SVM Results:")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    target_names = ['Negative', 'Neutral', 'Positive']
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("[[TN, FN, FP]")
    print(" [FN, TN, FP]")
    print(" [FP, FP, TN]]")
    print(cm)

    # Save the model and vectorizer
    joblib.dump(model, 'svm_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("\nModel saved as 'svm_model.pkl'")
    print("Vectorizer saved as 'tfidf_vectorizer.pkl'")

    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for new text."""
    # Transform text using the vectorizer
    text_tfidf = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_mapping[prediction]
    confidence = probabilities[prediction]

    return sentiment, confidence

if __name__ == "__main__":
    # Train and evaluate the model
    model, vectorizer = train_and_evaluate_model()

    # Interactive prediction
    print("\n" + "="*50)
    print("Airline Sentiment Analysis")
    print("="*50)
    print("Enter airline-related text to predict sentiment (or 'quit' to exit):")

    while True:
        user_input = input("\nText: ").strip()
        if user_input.lower() == 'quit':
            break

        if user_input:
            try:
                sentiment, confidence = predict_sentiment(user_input, model, vectorizer)
                print(f"Predicted sentiment: {sentiment} (Confidence: {confidence:.3f})")

            except Exception as e:
                print(f"Error making prediction: {e}")
        else:
            print("Please enter valid text.")