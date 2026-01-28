# ANN (Artificial Neural Network) Sentiment Analysis Model
# This script builds a sentiment classifier using a simple ANN on the IMDB dataset sample.

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
from sklearn.feature_extraction.text import TfidfVectorizer
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

class SentimentANN(nn.Module):
    def __init__(self, input_size):
        super(SentimentANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
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

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(preprocessed_reviews).toarray()
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model
    input_size = X_train.shape[1]
    model = SentimentANN(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("Training ANN model...")
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
    print(f"ANN Accuracy: {accuracy:.2f}")
    print(classification_report(all_labels, all_preds))

    # Prediction
    print("\nEnter a review to predict sentiment (or press enter to skip):")
    test_input = input().strip()
    if test_input:
        test_review = preprocess_text(test_input)
        test_vector = vectorizer.transform([test_review]).toarray()
        test_tensor = torch.tensor(test_vector, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            pred_prob = model(test_tensor)
            pred = 'positive' if pred_prob.item() > 0.5 else 'negative'
        print(f"Predicted sentiment: {pred}")