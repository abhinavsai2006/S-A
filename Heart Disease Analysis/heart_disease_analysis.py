# Heart Disease Analysis
# This script provides an overview of heart disease prediction using various ML models.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset."""
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

    return X_scaled, y, df

def analyze_dataset():
    """Analyze the heart disease dataset."""
    print("Heart Disease Dataset Analysis")
    print("=" * 40)

    X, y, df = load_and_preprocess_data()

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]}")

    print(f"\nClass distribution:")
    print(df['num'].value_counts())

    print(f"\nMissing values per column:")
    print(df.isnull().sum())

    print(f"\nFeature statistics:")
    print(df.describe())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

if __name__ == "__main__":
    analyze_dataset()

    print("\nAvailable models:")
    models = [
        'naive_bayes_model.py',
        'svm_model.py',
        'random_forest_model.py',
        'logistic_regression_model.py',
        'knn_model.py',
        'decision_tree_model.py',
        'ann_model.py',
        'cnn_model.py',
        'rnn_model.py',
        'lstm_model.py',
        'gru_model.py',
        'bilstm_model.py',
    ]

    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

    print("\nRun 'python model_comparison.py' to compare all models.")
    print("Run individual model files to train and test specific models.")