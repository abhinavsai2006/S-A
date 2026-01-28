# CNN Sentiment Analysis Model
# This script builds a sentiment classifier using CNN (Convolutional Neural Network) on the IMDB dataset sample.

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# GPU configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download required NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(SentimentCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv(x))
        x = self.pool(x).squeeze(2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def load_data(use_imdb=False):
    if use_imdb:
        # Load from IMDB Dataset.csv with sample
        df = pd.read_csv('IMDB Dataset.csv', nrows=5000)
        reviews = df['review'].tolist()
        labels = df['sentiment'].tolist()
        labels = [1 if l == 'positive' else 0 for l in labels]
    else:
        fileids = movie_reviews.fileids()
        reviews = []
        labels = []
        for fileid in fileids:
            review = movie_reviews.raw(fileid)
            reviews.append(review)
            if fileid.startswith('pos'):
                labels.append(1)  # positive
            else:
                labels.append(0)  # negative
    return reviews, labels

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

if __name__ == "__main__":
    print("Loading dataset...")
    reviews, labels = load_data(use_imdb=True)
    print(f"Loaded {len(reviews)} reviews")

    print("Preprocessing text...")
    preprocessed_reviews = [preprocess_text(review) for review in reviews]

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(preprocessed_reviews)
    sequences = tokenizer.texts_to_sequences(preprocessed_reviews)
    max_len = 200
    X = pad_sequences(sequences, maxlen=max_len)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model
    vocab_size = 5000
    embed_dim = 128
    model = SentimentCNN(vocab_size, embed_dim, max_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("Training CNN model...")
    model.train()
    for epoch in range(10):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/10, Loss: {loss.item():.4f}')

    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"CNN Accuracy: {accuracy:.2f}")
    print(classification_report(all_labels, all_preds))

    # Prediction
    print("\nEnter a review to predict sentiment (or press enter to skip):")
    test_input = input().strip()
    if test_input:
        test_review = preprocess_text(test_input)
        test_seq = tokenizer.texts_to_sequences([test_review])
        test_pad = pad_sequences(test_seq, maxlen=max_len)
        test_tensor = torch.tensor(test_pad, dtype=torch.long).to(device)
        model.eval()
        with torch.no_grad():
            pred_prob = model(test_tensor)
            pred = 'positive' if pred_prob.item() > 0.5 else 'negative'
        print(f"Predicted sentiment: {pred}")