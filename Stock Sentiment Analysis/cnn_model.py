# Stock Sentiment Analysis - CNN Model
# This script trains and tests a CNN model for stock sentiment classification.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=100):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab

    def build_vocab(self):
        """Build vocabulary from texts."""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for text in self.texts:
            for word in text.split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab

    def text_to_sequence(self, text):
        """Convert text to sequence of indices."""
        words = text.split()
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words[:self.max_len]]
        # Pad sequence
        if len(sequence) < self.max_len:
            sequence += [self.vocab['<PAD>']] * (self.max_len - len(sequence))
        return torch.tensor(sequence, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        sequence = self.text_to_sequence(text)

        return {
            'text': text,
            'sequence': sequence,
            'label': torch.tensor(label, dtype=torch.long)
        }

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, filter_sizes=[3, 4, 5], dropout=0.3):
        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 2)  # Binary classification

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)

        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        output = self.fc(cat)
        return output

def load_and_preprocess_data():
    """Load and preprocess the stock sentiment dataset."""
    # Load the dataset
    df = pd.read_csv('stock_data.csv')

    # Convert sentiment labels to 0 and 1
    df['Sentiment'] = df['Sentiment'].map({-1: 0, 1: 1})

    # Separate features and target
    X = df['Text'].values
    y = df['Sentiment'].values

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    """Create DataLoader objects for training and testing."""
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test, vocab=train_dataset.vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.vocab

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
    """Evaluate the model."""
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)

            outputs = model(sequences)
            _, preds = torch.max(outputs, dim=1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), all_preds, all_labels, all_probs

def train_and_evaluate_model():
    """Train and evaluate the CNN model."""
    print("Loading and preprocessing stock sentiment dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Create data loaders
    train_loader, test_loader, vocab = create_data_loaders(X_train, X_test, y_train, y_test)

    # Model parameters
    vocab_size = len(vocab)

    print("Training CNN model...")
    model = CNNClassifier(vocab_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

    # Evaluation
    test_acc, test_loss, y_pred, y_true, y_prob = eval_model(model, test_loader, criterion, device)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, [prob[1] for prob in y_prob])

    print(f"CNN Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print("[[TN, FP]")
    print(" [FN, TP]]")
    print(cm)

    # Save the model and vocabulary
    torch.save(model.state_dict(), 'cnn_model.pth')
    joblib.dump(vocab, 'cnn_vocab.pkl')
    print("\nModel saved as 'cnn_model.pth'")
    print("Vocabulary saved as 'cnn_vocab.pkl'")

    return model, vocab

def predict_sentiment(text, model, vocab, max_len=100):
    """Predict sentiment for new text."""
    model.eval()

    # Convert text to sequence
    words = text.split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words[:max_len]]
    if len(sequence) < max_len:
        sequence += [vocab['<PAD>']] * (max_len - len(sequence))

    sequence = torch.tensor([sequence], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(sequence)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, prediction = torch.max(outputs, dim=1)

    prediction = prediction.item()
    confidence = probs[0][prediction].item()

    sentiment = "Positive" if prediction == 1 else "Negative"

    return sentiment, confidence

if __name__ == "__main__":
    # Train and evaluate the model
    model, vocab = train_and_evaluate_model()

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
                sentiment, confidence = predict_sentiment(user_input, model, vocab)
                print(f"Predicted sentiment: {sentiment} (Confidence: {confidence:.3f})")

            except Exception as e:
                print(f"Error making prediction: {e}")
        else:
            print("Please enter valid text.")