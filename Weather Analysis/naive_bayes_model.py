# Naive Bayes Model for Weather Data Classification
# This script builds a classifier using Naive Bayes on the Combined12.csv dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

def load_data():
    # Load from Combined12.csv
    df = pd.read_csv('Combined12.csv')
    features = df.drop('normalized_label', axis=1)
    labels = df['normalized_label'].values
    return features.values, labels

# Load data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Naive Bayes Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))