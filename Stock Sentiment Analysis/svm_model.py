# Stock Sentiment Analysis - SVM Model
# This script trains and tests a Support Vector Machine model for stock sentiment classification.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the stock sentiment dataset."""
    # Load the dataset
    df = pd.read_csv('stock_data.csv')

    # Convert sentiment labels to 0 and 1
    df['Sentiment'] = df['Sentiment'].map({-1: 0, 1: 1})

    # Separate features and target
    X = df['Text']
    y = df['Sentiment']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def train_and_evaluate_model():
    """Train and evaluate the SVM model."""
    print("Loading and preprocessing stock sentiment dataset...")
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
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"SVM Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("[[TN, FP]")
    print(" [FN, TP]]")
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
    probability = model.predict_proba(text_tfidf)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[prediction]

    return sentiment, confidence

if __name__ == "__main__":
    # Train and evaluate the model
    model, vectorizer = train_and_evaluate_model()

    # Interactive prediction
    print("\n" + "="*50)
    print("Stock Sentiment Analysis")
    print("="*50)
    print("Enter stock-related text to predict sentiment (or 'quit' to exit):")

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