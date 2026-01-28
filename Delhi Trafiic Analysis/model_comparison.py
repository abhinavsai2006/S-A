# Delhi Traffic Analysis - Model Comparison Script
# This script compares all regression models for travel time prediction.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Set device for PyTorch models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import model classes for deep learning models
class ANNRegressor(nn.Module):
    def __init__(self, input_size):
        super(ANNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        return out

class CNNRegressor(nn.Module):
    def __init__(self, input_size):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        conv_output_size = input_size
        conv_output_size = (conv_output_size - 1) // 2
        conv_output_size = (conv_output_size - 1) // 2
        conv_output_size = (conv_output_size - 1) // 2

        self.fc1 = nn.Linear(128 * conv_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(RNNRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

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
    encoder = OneHotEncoder(sparse_output=False, drop='first')
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

    return X_train, X_test, y_train, y_test, scaler, encoder

def load_sklearn_model(model_name):
    """Load a scikit-learn model."""
    try:
        model = joblib.load(f'{model_name}_model.pkl')
        return model
    except FileNotFoundError:
        print(f"Warning: {model_name}_model.pkl not found. Skipping {model_name}.")
        return None

def load_pytorch_model(model_class, model_name, input_size):
    """Load a PyTorch model."""
    try:
        model = model_class(input_size).to(device)
        model.load_state_dict(torch.load(f'{model_name}_model.pth'))
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Warning: {model_name}_model.pth not found. Skipping {model_name}.")
        return None

def evaluate_model(model, X_test, y_test, model_type='sklearn', scaler=None, encoder=None):
    """Evaluate a model and return metrics."""
    if model_type == 'sklearn':
        y_pred = model.predict(X_test)
    else:  # PyTorch model
        X_tensor = torch.FloatTensor(X_test.values).to(device)
        with torch.no_grad():
            y_pred = model(X_tensor).cpu().numpy().flatten()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def main():
    """Main function to compare all models."""
    print("Delhi Traffic Analysis - Model Comparison")
    print("=" * 50)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess_data()

    # Model names and types
    sklearn_models = [
        ('linear_regression', 'Linear Regression'),
        ('ridge_regression', 'Ridge Regression'),
        ('lasso_regression', 'Lasso Regression'),
        ('random_forest_regressor', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('svm_regressor', 'SVM'),
        ('knn_regressor', 'KNN'),
        ('decision_tree_regressor', 'Decision Tree')
    ]

    pytorch_models = [
        ('ann_regressor', ANNRegressor, 'ANN'),
        ('cnn_regressor', CNNRegressor, 'CNN'),
        ('rnn_regressor', RNNRegressor, 'RNN'),
        ('lstm_regressor', LSTMRegressor, 'LSTM')
    ]

    results = []

    # Evaluate scikit-learn models
    print("\nEvaluating scikit-learn models...")
    for model_file, model_name in sklearn_models:
        model = load_sklearn_model(model_file)
        if model is not None:
            print(f"Evaluating {model_name}...")
            metrics = evaluate_model(model, X_test, y_test, 'sklearn')
            results.append({
                'Model': model_name,
                'MAE': metrics['MAE'],
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2']
            })

    # Evaluate PyTorch models
    print("\nEvaluating PyTorch models...")
    input_size = X_train.shape[1]
    for model_file, model_class, model_name in pytorch_models:
        model = load_pytorch_model(model_class, model_file, input_size)
        if model is not None:
            print(f"Evaluating {model_name}...")
            metrics = evaluate_model(model, X_test, y_test, 'pytorch')
            results.append({
                'Model': model_name,
                'MAE': metrics['MAE'],
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2']
            })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by R2 score (higher is better)
    results_df = results_df.sort_values('R2', ascending=False)

    # Display results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string(index=False, float_format='%.4f'))

    # Find best model
    best_model = results_df.iloc[0]
    print(f"\nBest Model: {best_model['Model']} (R² = {best_model['R2']:.4f})")

    # Save results to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")

    # Additional analysis
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS")
    print("="*50)

    # Group by model type
    sklearn_results = results_df[results_df['Model'].isin([name for _, name in sklearn_models])]
    pytorch_results = results_df[results_df['Model'].isin([name for _, _, name in pytorch_models])]

    print(f"Scikit-learn models average R²: {sklearn_results['R2'].mean():.4f}")
    print(f"PyTorch models average R²: {pytorch_results['R2'].mean():.4f}")

    # Best models in each category
    if not sklearn_results.empty:
        sklearn_best = sklearn_results.iloc[0]
        print(f"Best scikit-learn model: {sklearn_best['Model']} (R² = {sklearn_best['R2']:.4f})")

    if not pytorch_results.empty:
        pytorch_best = pytorch_results.iloc[0]
        print(f"Best PyTorch model: {pytorch_best['Model']} (R² = {pytorch_best['R2']:.4f})")

if __name__ == "__main__":
    main()