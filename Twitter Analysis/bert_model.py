# BERT Sentiment Analysis Model
# This script builds a sentiment classifier using BERT (Deep Learning) on the NLTK movie_reviews dataset.

import nltk
import torch
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd

# Download required NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(use_imdb=False):
    if use_imdb:
        # Load from training.1600000.processed.noemoticon.csv with sample
        df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, nrows=50000)
        reviews = df[5].tolist()
        labels = df[0].tolist()
        labels = [0 if l == 0 else 1 for l in labels]  # 0: negative, 4: positive
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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_reviews, labels, test_size=0.2, random_state=42)

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("Training BERT model...")
    trainer.train()

    # Evaluate
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"BERT Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))