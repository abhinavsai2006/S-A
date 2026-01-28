# Logistic Regression Heart Disease Prediction Model
# This script builds a heart disease classifier using Logistic Regression on the heart disease dataset.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data():
    # Load the heart disease dataset
    df = pd.read_csv('data.csv')

    # Clean column names
    df.columns = df.columns.str.strip()

    # Replace '?' with NaN
    df = df.replace('?', np.nan)

    # Separate features and target
    X = df.drop('num', axis=1)
    y = df['num']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

if __name__ == "__main__":
    print("Loading and preprocessing heart disease dataset...")
    X, y = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Prediction
    print("\nEnter patient features for prediction (comma-separated):")
    print("Format: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal")
    test_input = input().strip()
    if test_input:
        try:
            features = [float(x.strip()) for x in test_input.split(',')]
            features_scaled = (features - X.mean(axis=0)) / X.std(axis=0)  # Manual scaling
            prediction = model.predict([features_scaled])
            result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
            print(f"Prediction: {result}")
        except:
            print("Invalid input format. Please enter 13 numerical values separated by commas.")