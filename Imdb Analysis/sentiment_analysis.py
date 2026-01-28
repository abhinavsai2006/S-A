# Sentiment Analysis Application using IMDB Dataset
# This script builds a simple sentiment classifier that predicts if a movie review is Positive or Negative.
# It uses NLTK for text processing, scikit-learn for machine learning, and pandas for data handling.

# Step 1: Import necessary libraries
import nltk
import pandas as pd
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data (run this once if not already downloaded)
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Load the IMDB dataset
# The NLTK movie_reviews corpus contains 2000 movie reviews, 1000 positive and 1000 negative.
def load_data():
    # Get all file IDs
    fileids = movie_reviews.fileids()
    
    # Create lists for reviews and labels
    reviews = []
    labels = []
    
    for fileid in fileids:
        # Get the review text
        review = movie_reviews.raw(fileid)
        reviews.append(review)
        
        # Get the label (pos or neg)
        if fileid.startswith('pos'):
            labels.append('positive')
        else:
            labels.append('negative')
    
    return reviews, labels

print("Loading dataset...")
reviews, labels = load_data()
print(f"Loaded {len(reviews)} reviews: {labels.count('positive')} positive, {labels.count('negative')} negative")

# Step 3: Text Preprocessing
# We'll clean the text by converting to lowercase, removing HTML tags, special characters, stopwords, and lemmatizing.
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags (though IMDB reviews might not have many)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join back into string
    return ' '.join(tokens)

print("Preprocessing text...")
preprocessed_reviews = [preprocess_text(review) for review in reviews]
print("Preprocessing complete.")

# Step 4: Feature Extraction using TF-IDF
# TF-IDF converts text into numerical features based on word importance.
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features for simplicity
X = vectorizer.fit_transform(preprocessed_reviews)
y = labels

print(f"Feature extraction complete. Shape: {X.shape}")

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# Step 6: Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model training complete.")

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Prediction function for new text
def predict_sentiment(text):
    # Preprocess the input text
    processed_text = preprocess_text(text)
    
    # Vectorize
    vectorized_text = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(vectorized_text)[0]
    
    return prediction

# Example usage
if __name__ == "__main__":
    print("\n--- Sentiment Analysis Demo ---")
    sample_text = "This movie was absolutely amazing! The acting was superb and the plot kept me engaged."
    sentiment = predict_sentiment(sample_text)
    print(f"Sample text: '{sample_text}'")
    print(f"Predicted sentiment: {sentiment}")
    
    # Interactive example
    user_input = input("\nEnter a movie review to analyze: ")
    user_sentiment = predict_sentiment(user_input)
    print(f"Your review sentiment: {user_sentiment}")