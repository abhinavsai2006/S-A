# Airline Sentiment Analysis - Model Comparison
# This script compares all trained models for airline sentiment classification.

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import json
import warnings
warnings.filterwarnings('ignore')

# Import model classes
from bert_model import BERTClassifier
from lstm_model import LSTMClassifier
from cnn_model import CNNClassifier
from simple_nn_model import SimpleNNClassifier

def load_data():
    """Load and preprocess the airline sentiment dataset."""
    df = pd.read_csv('Tweets.csv')
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_label'] = df['airline_sentiment'].map(sentiment_mapping)

    X = df['text']
    y = df['sentiment_label']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_test, y_test

def evaluate_traditional_model(model_name, model_file, vectorizer_file, X_test, y_test):
    """Evaluate a traditional ML model."""
    try:
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)

        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        return {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred
        }
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return None

def evaluate_bert_model(X_test, y_test):
    """Evaluate BERT model."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BERTClassifier(n_classes=3)
        model.load_state_dict(torch.load('bert_model.pth'))
        model = model.to(device)
        model.eval()

        tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')

        predictions = []
        for text in X_test:
            encoding = tokenizer.encode_plus(
                str(text),
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

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.append(preds.item())

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')

        return {
            'model': 'BERT',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions
        }
    except Exception as e:
        print(f"Error evaluating BERT: {e}")
        return None

def evaluate_lstm_model(X_test, y_test):
    """Evaluate LSTM model."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('lstm_vocab.json', 'r') as f:
            vocab = json.load(f)

        model = LSTMClassifier(vocab_size=len(vocab), n_classes=3)
        model.load_state_dict(torch.load('lstm_model.pth'))
        model = model.to(device)
        model.eval()

        predictions = []
        max_len = 100

        for text in X_test:
            tokens = str(text).lower().split()
            indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]

            if len(indices) < max_len:
                indices += [vocab['<PAD>']] * (max_len - len(indices))
            else:
                indices = indices[:max_len]

            input_ids = torch.tensor([indices], dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model(input_ids)
                _, preds = torch.max(outputs, dim=1)
                predictions.append(preds.item())

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')

        return {
            'model': 'LSTM',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions
        }
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")
        return None

def evaluate_simple_nn_model(X_test, y_test):
    """Evaluate Simple NN model."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('simple_nn_vocab.json', 'r') as f:
            vocab = json.load(f)

        model = SimpleNNClassifier(vocab_size=len(vocab), n_classes=3)
        model.load_state_dict(torch.load('simple_nn_model.pth'))
        model = model.to(device)
        model.eval()

        predictions = []
        max_len = 100

        for text in X_test:
            tokens = str(text).lower().split()
            indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]

            if len(indices) < max_len:
                indices += [vocab['<PAD>']] * (max_len - len(indices))
            else:
                indices = indices[:max_len]

            input_ids = torch.tensor([indices], dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model(input_ids)
                _, preds = torch.max(outputs, dim=1)
                predictions.append(preds.item())

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')

        return {
            'model': 'Simple NN',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions
        }
    except Exception as e:
        print(f"Error evaluating Simple NN: {e}")
        return None

def evaluate_cnn_model(X_test, y_test):
    """Evaluate CNN model."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('cnn_vocab.json', 'r') as f:
            vocab = json.load(f)

        model = CNNClassifier(vocab_size=len(vocab), n_classes=3)
        model.load_state_dict(torch.load('cnn_model.pth'))
        model = model.to(device)
        model.eval()

        predictions = []
        max_len = 100

        for text in X_test:
            tokens = str(text).lower().split()
            indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]

            if len(indices) < max_len:
                indices += [vocab['<PAD>']] * (max_len - len(indices))
            else:
                indices = indices[:max_len]

            input_ids = torch.tensor([indices], dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model(input_ids)
                _, preds = torch.max(outputs, dim=1)
                predictions.append(preds.item())

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')

        return {
            'model': 'CNN',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions
        }
    except Exception as e:
        print(f"Error evaluating CNN: {e}")
        return None

def create_comparison_table(results):
    """Create a comparison table of all model results."""
    print("\n" + "="*80)
    print("AIRLINE SENTIMENT ANALYSIS - MODEL COMPARISON")
    print("="*80)

    # Create DataFrame for easy display
    df_results = pd.DataFrame(results)
    df_results = df_results.round(4)

    print("\nPERFORMANCE METRICS:")
    print("-" * 80)
    print(df_results.to_string(index=False))

    # Find best models
    best_accuracy = df_results.loc[df_results['accuracy'].idxmax()]
    best_f1 = df_results.loc[df_results['f1_score'].idxmax()]

    print(f"\n🏆 BEST ACCURACY: {best_accuracy['model']} ({best_accuracy['accuracy']:.4f})")
    print(f"🏆 BEST F1-SCORE: {best_f1['model']} ({best_f1['f1_score']:.4f})")

    return df_results

def detailed_classification_reports(results, y_test):
    """Generate detailed classification reports for each model."""
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*80)

    target_names = ['Negative', 'Neutral', 'Positive']

    for result in results:
        if result is None:
            continue

        print(f"\n{result['model']} Model:")
        print("-" * 40)
        print(classification_report(y_test, result['predictions'], target_names=target_names))

def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all models."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_models = len([r for r in results if r is not None])
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()

    target_names = ['Negative', 'Neutral', 'Positive']

    for i, result in enumerate([r for r in results if r is not None]):
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names, ax=axes[i])
        axes[i].set_title(f'{result["model"]} Confusion Matrix')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('model_comparison_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nConfusion matrices saved as 'model_comparison_confusion_matrices.png'")

def main():
    """Main function to run model comparison."""
    print("Loading test data...")
    X_test, y_test = load_data()

    print("Evaluating models...")

    # Evaluate all models
    results = []

    # Traditional ML models
    traditional_models = [
        ('Logistic Regression', 'logistic_regression_model.pkl', 'tfidf_vectorizer.pkl'),
        ('Naive Bayes', 'naive_bayes_model.pkl', 'tfidf_vectorizer.pkl'),
        ('SVM', 'svm_model.pkl', 'tfidf_vectorizer.pkl'),
        ('Random Forest', 'random_forest_model.pkl', 'tfidf_vectorizer.pkl'),
        ('Gradient Boosting', 'gradient_boosting_model.pkl', 'tfidf_vectorizer.pkl')
    ]

    for model_name, model_file, vectorizer_file in traditional_models:
        result = evaluate_traditional_model(model_name, model_file, vectorizer_file, X_test, y_test)
        if result:
            results.append(result)

    # Deep Learning models
    bert_result = evaluate_bert_model(X_test, y_test)
    if bert_result:
        results.append(bert_result)

    lstm_result = evaluate_lstm_model(X_test, y_test)
    if lstm_result:
        results.append(lstm_result)

    simple_nn_result = evaluate_simple_nn_model(X_test, y_test)
    if simple_nn_result:
        results.append(simple_nn_result)

    cnn_result = evaluate_cnn_model(X_test, y_test)
    if cnn_result:
        results.append(cnn_result)

    if not results:
        print("No models could be evaluated. Please check if model files exist.")
        return

    # Create comparison table
    df_results = create_comparison_table(results)

    # Detailed reports
    detailed_classification_reports(results, y_test)

    # Plot confusion matrices
    try:
        plot_confusion_matrices(results, y_test)
    except ImportError:
        print("\nMatplotlib and seaborn not available for plotting confusion matrices.")

    # Save results to CSV
    df_results.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")

    print("\n" + "="*80)
    print("MODEL COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()