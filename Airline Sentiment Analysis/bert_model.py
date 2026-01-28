# Airline Sentiment Analysis - BERT Model
# This script trains and tests a BERT model for airline sentiment classification.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

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

        encoding = self.tokenizer(
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

class BERTClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super(BERTClassifier, self).__init__()
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

def train_model(model, data_loader, optimizer, scheduler, device):
    """Train the BERT model for one epoch."""
    model.train()
    losses = []
    correct_predictions = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    """Evaluate the BERT model."""
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def train_and_evaluate_model():
    """Train and evaluate the BERT model."""
    print("Loading and preprocessing airline sentiment dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset = SentimentDataset(X_train.values, y_train.values, tokenizer)
    test_dataset = SentimentDataset(X_test.values, y_test.values, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERTClassifier(n_classes=3)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3  # 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop
    print("Training BERT model...")
    best_accuracy = 0

    for epoch in range(3):
        print(f'Epoch {epoch + 1}/3')
        train_acc, train_loss = train_model(model, train_loader, optimizer, scheduler, device)
        print(f'Train loss: {train_loss:.3f}, Train accuracy: {train_acc:.3f}')

        val_acc, val_loss = eval_model(model, test_loader, device)
        print(f'Val loss: {val_loss:.3f}, Val accuracy: {val_acc:.3f}')

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'bert_model.pth')

    # Load best model
    model.load_state_dict(torch.load('bert_model.pth'))

    # Final evaluation
    print("\nEvaluating final model...")
    model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for data in test_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(real_values, predictions)

    print(f"BERT Results:")
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

    # Save tokenizer
    tokenizer.save_pretrained('bert_tokenizer')

    print("\nModel saved as 'bert_model.pth'")
    print("Tokenizer saved in 'bert_tokenizer' directory")

    return model, tokenizer, device

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for new text."""
    model.eval()

    encoding = tokenizer(
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
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_mapping[prediction.item()]
    confidence = probabilities[0][prediction.item()].item()

    return sentiment, confidence

if __name__ == "__main__":
    # Train and evaluate the model
    model, tokenizer, device = train_and_evaluate_model()

    # Interactive prediction
    print("\n" + "="*50)
    print("Airline Sentiment Analysis - BERT")
    print("="*50)
    print("Enter airline-related text to predict sentiment (or 'quit' to exit):")

    while True:
        user_input = input("\nText: ").strip()
        if user_input.lower() == 'quit':
            break

        if user_input:
            try:
                sentiment, confidence = predict_sentiment(user_input, model, tokenizer, device)
                print(f"Predicted sentiment: {sentiment} (Confidence: {confidence:.3f})")

            except Exception as e:
                print(f"Error making prediction: {e}")
        else:
            print("Please enter valid text.")