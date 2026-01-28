# Stock Sentiment Analysis - BERT Model
# This script trains and tests a BERT model for stock sentiment classification.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

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

def create_data_loaders(X_train, X_test, y_train, y_test, tokenizer, batch_size=16):
    """Create DataLoader objects for training and testing."""
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    """Train the model for one epoch."""
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
    """Evaluate the model."""
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
    """Train and evaluate the BERT model."""
    print("Loading and preprocessing stock sentiment dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test, tokenizer)

    # Initialize model
    model = BERTSentimentClassifier(n_classes=2)
    model = model.to(device)

    # Training parameters
    epochs = 3
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    print("Training BERT model...")
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scheduler
        )

        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

    # Evaluation
    test_acc, test_loss, y_pred, y_true, y_prob = eval_model(
        model, test_loader, loss_fn, device
    )

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, [prob[1] for prob in y_prob])

    print(f"BERT Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print("[[TN, FP]")
    print(" [FN, TP]]")
    print(cm)

    # Save the model and tokenizer
    torch.save(model.state_dict(), 'bert_model.pth')
    tokenizer.save_pretrained('./bert_tokenizer')
    print("\nModel saved as 'bert_model.pth'")
    print("Tokenizer saved in './bert_tokenizer'")

    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for new text."""
    model.eval()

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

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, prediction = torch.max(outputs, dim=1)

    prediction = prediction.item()
    confidence = probs[0][prediction].item()

    sentiment = "Positive" if prediction == 1 else "Negative"

    return sentiment, confidence

if __name__ == "__main__":
    # Train and evaluate the model
    model, tokenizer = train_and_evaluate_model()

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
                sentiment, confidence = predict_sentiment(user_input, model, tokenizer)
                print(f"Predicted sentiment: {sentiment} (Confidence: {confidence:.3f})")

            except Exception as e:
                print(f"Error making prediction: {e}")
        else:
            print("Please enter valid text.")