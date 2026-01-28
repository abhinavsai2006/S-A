# Decision Tree Sentiment Analysis Model
# This script builds a sentiment classifier using Decision Tree on the IMDB dataset sample.

import nltk
import pandas as pd
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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
                labels.append('positive')
            else:
                labels.append('negative')
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
    X = vectorizer.fit_transform(preprocessed_reviews)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))