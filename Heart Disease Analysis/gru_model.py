# GRU Heart Disease Prediction Model
# This script builds a heart disease classifier using GRU (Gated Recurrent Unit) on the heart disease dataset.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# GPU configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class HeartDiseaseGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HeartDiseaseGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension: (batch, 1, features)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take last output
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))
        return out

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

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model
    input_size = X_train.shape[1]
    hidden_size = 128
    model = HeartDiseaseGRU(input_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("Training GRU model...")
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
    print(f"GRU Accuracy: {accuracy:.2f}")
    print(classification_report(all_labels, all_preds))

    # Prediction
    print("\nEnter patient features for prediction (comma-separated):")
    print("Format: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal")
    test_input = input().strip()
    if test_input:
        try:
            features = [float(x.strip()) for x in test_input.split(',')]
            features_scaled = (features - X.mean(axis=0)) / X.std(axis=0)  # Manual scaling
            features_tensor = torch.tensor([features_scaled], dtype=torch.float32).to(device)
            model.eval()
            with torch.no_grad():
                pred_prob = model(features_tensor)
                pred = 'Heart Disease' if pred_prob.item() > 0.5 else 'No Heart Disease'
            print(f"Prediction: {pred}")
        except:
            print("Invalid input format. Please enter 13 numerical values separated by commas.")