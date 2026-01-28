# Delhi Traffic Analysis
# This script provides an overview of Delhi traffic travel time prediction using various ML models.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the Delhi traffic dataset."""
    # Load the datasets
    df_features = pd.read_csv('delhi_traffic_features.csv')
    df_target = pd.read_csv('delhi_traffic_target.csv')

    # Clean column names
    df_features.columns = df_features.columns.str.strip()
    df_target.columns = df_target.columns.str.strip()

    # Merge datasets on Trip_ID
    df = pd.merge(df_features, df_target, on='Trip_ID', how='inner')

    print("Loading and preprocessing Delhi traffic dataset...")
    print(f"Dataset shape: {df.shape}")

    # Drop Trip_ID as it's not a feature
    df = df.drop('Trip_ID', axis=1)

    # Separate features and target
    X = df.drop('travel_time_minutes', axis=1)
    y = df['travel_time_minutes']

    # Identify categorical and numerical columns
    categorical_cols = ['start_area', 'end_area', 'time_of_day', 'day_of_week',
                       'weather_condition', 'traffic_density_level', 'road_type']
    numerical_cols = ['distance_km', 'average_speed_kmph']

    # Handle categorical variables - One-hot encoding
    encoder = OneHotEncoder(sparse=False, drop='first')
    X_categorical = pd.DataFrame(
        encoder.fit_transform(X[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Handle numerical variables - Standard scaling
    scaler = StandardScaler()
    X_numerical = pd.DataFrame(
        scaler.fit_transform(X[numerical_cols]),
        columns=numerical_cols
    )

    # Combine processed features
    X_processed = pd.concat([X_numerical, X_categorical], axis=1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature vector size: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler, encoder, df

def analyze_dataset():
    """Analyze the Delhi traffic dataset."""
    X_train, X_test, y_train, y_test, scaler, encoder, df = load_and_preprocess_data()

    print("\nDelhi Traffic Dataset Analysis")
    print("=" * 50)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")

    print("\nTarget variable statistics:")
    print(df['travel_time_minutes'].describe())

    print("\nCategorical feature distributions:")
    categorical_cols = ['time_of_day', 'day_of_week', 'weather_condition',
                       'traffic_density_level', 'road_type']
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print("\nAvailable models:")
    models = [
        'linear_regression_model.py',
        'ridge_regression_model.py',
        'lasso_regression_model.py',
        'random_forest_regressor_model.py',
        'gradient_boosting_model.py',
        'svm_regressor_model.py',
        'knn_regressor_model.py',
        'decision_tree_regressor_model.py',
        'ann_regressor_model.py',
        'cnn_regressor_model.py',
        'rnn_regressor_model.py',
        'lstm_regressor_model.py',
    ]

    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

    print("\nRun 'python model_comparison.py' to compare all models.")
    print("Run individual model files to train and test specific models.")

if __name__ == "__main__":
    analyze_dataset()