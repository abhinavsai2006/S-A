# SVM Model for Weather Data Classification
# This script builds a classifier using SVM on the Combined12.csv dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_data():
    # Load from Combined12.csv
    df = pd.read_csv('Combined12.csv')
    features = df.drop('normalized_label', axis=1)
    labels = df['normalized_label'].values
    return features.values, labels

# Load data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train model
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Prediction
print("\nEnter 7 features separated by space to predict (or press enter to skip):")
test_input = input().strip()
if test_input:
    features = list(map(float, test_input.split()))
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)
    print(f"Predicted: {pred[0]}")