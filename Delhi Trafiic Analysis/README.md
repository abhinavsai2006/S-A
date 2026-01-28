# Delhi Traffic Analysis

A comprehensive machine learning system for predicting travel time in Delhi traffic using multiple regression models.

## Dataset

The system uses Delhi traffic data with the following features:
- **Trip_ID**: Unique identifier for each trip
- **start_area**: Starting location/area in Delhi
- **end_area**: Destination location/area in Delhi
- **distance_km**: Distance of the trip in kilometers
- **time_of_day**: Time period (Morning Peak, Evening Peak, Off-Peak, Night)
- **day_of_week**: Day of the week (Weekday, Weekend)
- **weather_condition**: Weather conditions (Clear, Rain, Fog, Storm)
- **traffic_density_level**: Traffic density (Low, Medium, High)
- **road_type**: Type of road (Main Road, Highway, Inner Road)
- **average_speed_kmph**: Average speed in km/h
- **travel_time_minutes**: Target variable - travel time in minutes

## Models Implemented

### Traditional Machine Learning Models (Scikit-learn)
1. **Linear Regression** (`linear_regression_model.py`)
2. **Ridge Regression** (`ridge_regression_model.py`)
3. **Lasso Regression** (`lasso_regression_model.py`)
4. **Random Forest Regressor** (`random_forest_regressor_model.py`)
5. **Gradient Boosting Regressor** (`gradient_boosting_model.py`)
6. **Support Vector Regressor** (`svm_regressor_model.py`)
7. **K-Nearest Neighbors Regressor** (`knn_regressor_model.py`)
8. **Decision Tree Regressor** (`decision_tree_regressor_model.py`)

### Deep Learning Models (PyTorch)
1. **Artificial Neural Network** (`ann_regressor_model.py`)
2. **Convolutional Neural Network** (`cnn_regressor_model.py`)
3. **Recurrent Neural Network** (`rnn_regressor_model.py`)
4. **Long Short-Term Memory** (`lstm_regressor_model.py`)

## Data Preprocessing

- **Categorical Encoding**: One-hot encoding for categorical variables
- **Numerical Scaling**: Standard scaling for numerical features
- **Train/Test Split**: 80/20 split with random state 42

## Usage

### Training Individual Models

Run any model file to train and evaluate:

```bash
python linear_regression_model.py
python ann_regressor_model.py
# etc.
```

### Model Comparison

Run the comparison script to evaluate all trained models:

```bash
python model_comparison.py
```

### Interactive Prediction

After training, each model provides an interactive interface for making predictions. Enter trip features in the format:

```
start_area,end_area,distance_km,time_of_day,day_of_week,weather_condition,traffic_density_level,road_type,average_speed_kmph
```

Example:
```
Greater Kailash,IGI Airport,13.13,Morning Peak,Weekend,Clear,Low,Main Road,58.1
```

## Model Files

- `{model_name}_model.pkl`: Trained scikit-learn models
- `{model_name}_model.pth`: Trained PyTorch models
- `scaler.pkl`: Standard scaler for numerical features
- `encoder.pkl`: One-hot encoder for categorical features
- `model_comparison_results.csv`: Comparison results of all models

## Performance Metrics

Models are evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- PyTorch
- joblib

## Installation

```bash
pip install pandas numpy scikit-learn torch joblib
```

## Project Structure

```
Delhi Trafiic Analysis/
├── delhi_traffic_analysis.py          # Main analysis file
├── linear_regression_model.py         # Linear Regression model
├── ridge_regression_model.py          # Ridge Regression model
├── lasso_regression_model.py          # Lasso Regression model
├── random_forest_regressor_model.py   # Random Forest model
├── gradient_boosting_model.py         # Gradient Boosting model
├── svm_regressor_model.py             # SVM model
├── knn_regressor_model.py             # KNN model
├── decision_tree_regressor_model.py   # Decision Tree model
├── ann_regressor_model.py             # ANN model
├── cnn_regressor_model.py             # CNN model
├── rnn_regressor_model.py             # RNN model
├── lstm_regressor_model.py            # LSTM model
├── model_comparison.py                # Model comparison script
├── README.md                          # This file
├── scaler.pkl                         # Saved scaler
├── encoder.pkl                        # Saved encoder
├── *_model.pkl                        # Trained sklearn models
├── *_model.pth                        # Trained PyTorch models
└── model_comparison_results.csv       # Comparison results
```

## Notes

- All models use the same preprocessing pipeline for fair comparison
- Deep learning models are adapted for tabular data
- Interactive prediction is available for all trained models
- Model comparison automatically detects and evaluates available trained models