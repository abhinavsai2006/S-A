# Stock Sentiment Analysis - Model Comparison Script
# This script compares all sentiment analysis models.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Set device for PyTorch models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import model classes for deep learning models
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, filter_sizes=[3, 4, 5], dropout=0.3):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        output = self.fc(cat)
        return output

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, n_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        output = self.fc(hidden)
        return output

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

def load_sklearn_model(model_name):
    """Load a scikit-learn model."""
    try:
        model = joblib.load(f'{model_name}_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        print(f"Warning: {model_name}_model.pkl or tfidf_vectorizer.pkl not found. Skipping {model_name}.")
        return None, None

def load_pytorch_model(model_class, model_name, vocab_file):
    """Load a PyTorch model."""
    try:
        vocab = joblib.load(vocab_file)
        vocab_size = len(vocab)
        model = model_class(vocab_size).to(device)
        model.load_state_dict(torch.load(f'{model_name}_model.pth'))
        model.eval()
        return model, vocab
    except FileNotFoundError:
        print(f"Warning: {model_name}_model.pth or {vocab_file} not found. Skipping {model_name}.")
        return None, None

def evaluate_sklearn_model(model, vectorizer, X_test, y_test):
    """Evaluate a scikit-learn model."""
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, auc

def evaluate_pytorch_model(model, vocab, X_test, y_test, max_len=100):
    """Evaluate a PyTorch model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for text, label in zip(X_test, y_test):
            # Convert text to sequence
            words = text.split()
            sequence = [vocab.get(word, vocab['<UNK>']) for word in words[:max_len]]
            if len(sequence) < max_len:
                sequence += [vocab['<PAD>']] * (max_len - len(sequence))

            sequence = torch.tensor([sequence], dtype=torch.long).to(device)
            outputs = model(sequence)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, prediction = torch.max(outputs, dim=1)

            all_preds.append(prediction.item())
            all_labels.append(label)
            all_probs.append(probs[0][1].item())

    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return accuracy, auc

def main():
    """Main function to compare all models."""
    print("Stock Sentiment Analysis - Model Comparison")
    print("=" * 50)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Model names and types
    sklearn_models = [
        ('logistic_regression', 'Logistic Regression'),
        ('naive_bayes', 'Naive Bayes'),
        ('svm', 'SVM'),
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting')
    ]

    pytorch_models = [
        ('cnn', CNNClassifier, 'cnn_vocab.pkl', 'CNN'),
        ('lstm', LSTMClassifier, 'lstm_vocab.pkl', 'LSTM')
    ]

    results = []

    # Evaluate scikit-learn models
    print("\nEvaluating scikit-learn models...")
    for model_file, model_name in sklearn_models:
        model, vectorizer = load_sklearn_model(model_file)
        if model is not None:
            print(f"Evaluating {model_name}...")
            accuracy, auc = evaluate_sklearn_model(model, vectorizer, X_test, y_test)
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'AUC': auc
            })

    # Evaluate PyTorch models
    print("\nEvaluating PyTorch models...")
    for model_file, model_class, vocab_file, model_name in pytorch_models:
        model, vocab = load_pytorch_model(model_class, model_file, vocab_file)
        if model is not None:
            print(f"Evaluating {model_name}...")
            accuracy, auc = evaluate_pytorch_model(model, vocab, X_test, y_test)
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'AUC': auc
            })

    # Check for BERT model separately
    try:
        from transformers import BertTokenizer, BertModel
        import torch.nn as nn

        class BERTSentimentClassifier(nn.Module):
            def __init__(self, n_classes=2):
                super(BERTSentimentClassifier, self).__init__()
                self.bert = BertModel.from_pretrained('bert-base-uncased')
                self.drop = nn.Dropout(p=0.3)
                self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

            def forward(self, input_ids, attention_mask):
                _, pooled_output = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=False
                )
                output = self.drop(pooled_output)
                return self.out(output)

        tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer')
        model = BERTSentimentClassifier(n_classes=2)
        model.load_state_dict(torch.load('bert_model.pth'))
        model = model.to(device)
        model.eval()

        print("Evaluating BERT...")
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for text, label in zip(X_test, y_test):
                encoding = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, prediction = torch.max(outputs, dim=1)

                all_preds.append(prediction.item())
                all_labels.append(label)
                all_probs.append(probs[0][1].item())

        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        results.append({
            'Model': 'BERT',
            'Accuracy': accuracy,
            'AUC': auc
        })

    except Exception as e:
        print(f"Warning: BERT model evaluation failed: {e}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by Accuracy (higher is better)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    # Display results
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(results_df.to_string(index=False, float_format='%.4f'))

    # Find best model
    best_model = results_df.iloc[0]
    print(f"\nBest Model: {best_model['Model']} (Accuracy = {best_model['Accuracy']:.4f})")

    # Save results to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")

    # Additional analysis
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS")
    print("="*50)

    # Group by model type
    sklearn_results = results_df[results_df['Model'].isin([name for _, name in sklearn_models])]
    pytorch_results = results_df[results_df['Model'].isin([name for _, _, _, name in pytorch_models] + ['BERT'])]

    if not sklearn_results.empty:
        print(f"Scikit-learn models average accuracy: {sklearn_results['Accuracy'].mean():.4f}")
        sklearn_best = sklearn_results.iloc[0]
        print(f"Best scikit-learn model: {sklearn_best['Model']} (Accuracy = {sklearn_best['Accuracy']:.4f})")

    if not pytorch_results.empty:
        print(f"PyTorch models average accuracy: {pytorch_results['Accuracy'].mean():.4f}")
        pytorch_best = pytorch_results.iloc[0]
        print(f"Best PyTorch model: {pytorch_best['Model']} (Accuracy = {pytorch_best['Accuracy']:.4f})")

if __name__ == "__main__":
    main()