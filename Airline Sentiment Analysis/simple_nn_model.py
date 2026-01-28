# Airline Sentiment Analysis - Simple NN Model
# This script trains and tests a simple neural network for airline sentiment classification.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize and convert to indices
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

        # Pad or truncate
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return {
            'text': text,
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SimpleNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, n_classes=3):
        super(SimpleNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(embedding_dim * 100, hidden_dim)  # 100 is max_len
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        flattened = self.flatten(embedded)    # [batch_size, seq_len * embedding_dim]
        out = self.fc1(flattened)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def build_vocab(texts, max_vocab_size=10000):
    """Build vocabulary from texts."""
    from collections import Counter

    # Count word frequencies
    counter = Counter()
    for text in texts:
        tokens = str(text).lower().split()
        counter.update(tokens)

    # Create vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)

    return vocab

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

def train_model(model, data_loader, optimizer, criterion, device):
    """Train the Simple NN model for one epoch."""
    model.train()
    losses = []
    correct_predictions = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, criterion, device):
    """Evaluate the Simple NN model."""
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            labels = data['label'].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def train_and_evaluate_model():
    """Train and evaluate the Simple NN model."""
    print("Loading and preprocessing airline sentiment dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(X_train)

    # Create datasets
    train_dataset = SentimentDataset(X_train.values, y_train.values, vocab)
    test_dataset = SentimentDataset(X_test.values, y_test.values, vocab)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNNClassifier(vocab_size=len(vocab), n_classes=3)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Training Simple NN model...")
    best_accuracy = 0

    for epoch in range(10):
        print(f'Epoch {epoch + 1}/10')
        train_acc, train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Train loss: {train_loss:.3f}, Train accuracy: {train_acc:.3f}')

        val_acc, val_loss = eval_model(model, test_loader, criterion, device)
        print(f'Val loss: {val_loss:.3f}, Val accuracy: {val_acc:.3f}')

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'simple_nn_model.pth')

    # Load best model
    model.load_state_dict(torch.load('simple_nn_model.pth'))

    # Final evaluation
    print("\nEvaluating final model...")
    model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for data in test_loader:
            input_ids = data['input_ids'].to(device)
            labels = data['label'].to(device)

            outputs = model(input_ids)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(real_values, predictions)

    print(f"Simple NN Results:")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    target_names = ['Negative', 'Neutral', 'Positive']
    print(classification_report(real_values, predictions, target_names=target_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(real_values, predictions)
    print("[[TN, FN, FP]")
    print(" [FN, TN, FP]")
    print(" [FP, FP, TN]]")
    print(cm)

    # Save vocabulary
    import json
    with open('simple_nn_vocab.json', 'w') as f:
        json.dump(vocab, f)

    print("\nModel saved as 'simple_nn_model.pth'")
    print("Vocabulary saved as 'simple_nn_vocab.json'")

    return model, vocab, device

def predict_sentiment(text, model, vocab, device, max_len=100):
    """Predict sentiment for new text."""
    model.eval()

    # Tokenize and convert to indices
    tokens = text.lower().split()
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]

    # Pad or truncate
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    input_ids = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_mapping[prediction.item()]
    confidence = probabilities[0][prediction.item()].item()

    return sentiment, confidence

if __name__ == "__main__":
    # Train and evaluate the model
    model, vocab, device = train_and_evaluate_model()

    # Interactive prediction
    print("\n" + "="*50)
    print("Airline Sentiment Analysis - Simple NN")
    print("="*50)
    print("Enter airline-related text to predict sentiment (or 'quit' to exit):")

    while True:
        user_input = input("\nText: ").strip()
        if user_input.lower() == 'quit':
            break

        if user_input:
            try:
                sentiment, confidence = predict_sentiment(user_input, model, vocab, device)
                print(f"Predicted sentiment: {sentiment} (Confidence: {confidence:.3f})")

            except Exception as e:
                print(f"Error making prediction: {e}")
        else:
            print("Please enter valid text.")