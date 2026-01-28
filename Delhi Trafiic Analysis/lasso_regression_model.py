# Delhi Traffic Analysis - Lasso Regression Model
# This script trains and tests a Lasso Regression model for travel time prediction.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
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
    """Train and evaluate the Lasso Regression model."""
    print("Loading and preprocessing Delhi traffic dataset...")
    X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess_data()

    print("Training Lasso Regression model...")
    # Create and train the model
    model = Lasso(alpha=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Lasso Regression Results:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Save the model and preprocessors
    joblib.dump(model, 'lasso_regression_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    print("\nModel saved as 'lasso_regression_model.pkl'")

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

    # Make prediction
    prediction = model.predict(X_processed)[0]

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