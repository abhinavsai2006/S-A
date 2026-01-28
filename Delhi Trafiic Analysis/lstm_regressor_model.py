# Delhi Traffic Analysis - LSTM Regressor Model (PyTorch)
# This script trains and tests a Long Short-Term Memory Regressor for travel time prediction.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # For tabular data, we treat each sample as a sequence of length 1
        # Reshape to (batch_size, seq_len=1, input_size)
        x = x.unsqueeze(1)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Fully connected layers
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

def train_and_evaluate_model():
    """Train and evaluate the LSTM Regressor model."""
    print("Loading and preprocessing Delhi traffic dataset...")
    X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess_data()

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model parameters
    input_size = X_train.shape[1]

    print("Training LSTM Regressor model...")
    model = LSTMRegressor(input_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = []
        for inputs, _ in test_loader:
            outputs = model(inputs)
            y_pred.extend(outputs.cpu().numpy().flatten())

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"LSTM Regressor Results:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Save the model and preprocessors
    torch.save(model.state_dict(), 'lstm_regressor_model.pth')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    print("\nModel saved as 'lstm_regressor_model.pth'")

    return model, scaler, encoder

def predict_travel_time(features, model, scaler, encoder):
    """Predict travel time for new trip features."""
    # Convert features to DataFrame if it's a dict
    if isinstance(features, dict):
        features = pd.DataFrame([features])

    # Identify categorical and numerical columns
    categorical_cols = ['start_area', 'end_area', 'time_of_day', 'day_of_week',
                       'weather_condition', 'traffic_density_level', 'road_type']
    numerical_cols = ['distance_km', 'average_speed_kmph']

    # Handle categorical variables
    X_categorical = pd.DataFrame(
        encoder.transform(features[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Handle numerical variables
    X_numerical = pd.DataFrame(
        scaler.transform(features[numerical_cols]),
        columns=numerical_cols
    )

    # Combine processed features
    X_processed = pd.concat([X_numerical, X_categorical], axis=1)

    # Convert to tensor
    X_tensor = torch.FloatTensor(X_processed.values).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(X_tensor).cpu().numpy()[0][0]

    return prediction

if __name__ == "__main__":
    # Train and evaluate the model
    model, scaler, encoder = train_and_evaluate_model()

    # Interactive prediction
    print("\n" + "="*50)
    print("Delhi Traffic Travel Time Prediction")
    print("="*50)
    print("Enter trip features to predict travel time (or 'quit' to exit):")
    print("Format: start_area,end_area,distance_km,time_of_day,day_of_week,weather_condition,traffic_density_level,road_type,average_speed_kmph")

    while True:
        user_input = input("\nTrip features (comma-separated): ").strip()
        if user_input.lower() == 'quit':
            break

        if user_input:
            try:
                # Parse input
                parts = [part.strip() for part in user_input.split(',')]
                if len(parts) != 9:
                    print("Please enter exactly 9 features separated by commas.")
                    continue

                features = {
                    'start_area': parts[0],
                    'end_area': parts[1],
                    'distance_km': float(parts[2]),
                    'time_of_day': parts[3],
                    'day_of_week': parts[4],
                    'weather_condition': parts[5],
                    'traffic_density_level': parts[6],
                    'road_type': parts[7],
                    'average_speed_kmph': float(parts[8])
                }

                predicted_time = predict_travel_time(features, model, scaler, encoder)
                print(f"Predicted travel time: {predicted_time:.1f} minutes")

            except ValueError as e:
                print(f"Invalid input format. Please enter numerical values for distance and speed. Error: {e}")
            except Exception as e:
                print(f"Error making prediction: {e}")
        else:
            print("Please enter valid trip features.")