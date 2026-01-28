# Stock Sentiment Analysis

A comprehensive machine learning system for analyzing stock market sentiment from text data using multiple classification models.

## Dataset

The system uses stock sentiment data with the following features:
- **Text**: Stock-related text content (tweets, comments, news snippets)
- **Sentiment**: Binary classification (-1 for negative/bearish, 1 for positive/bullish)

## Dataset Statistics

- **Total samples**: 5,791
- **Positive sentiments (1)**: 3,685 (63.6%)
- **Negative sentiments (-1)**: 2,106 (36.4%)
- **Average text length**: ~100 characters
- **Text preprocessing**: TF-IDF vectorization for traditional ML, custom tokenization for deep learning

## Models Implemented

### Traditional Machine Learning Models (Scikit-learn)
1. **Logistic Regression** (`logistic_regression_model.py`)
2. **Naive Bayes** (`naive_bayes_model.py`)
3. **Support Vector Machine** (`svm_model.py`)
4. **Random Forest** (`random_forest_model.py`)
5. **Gradient Boosting** (`gradient_boosting_model.py`)

### Deep Learning Models (PyTorch)
1. **BERT Transformer** (`bert_model.py`) - State-of-the-art transformer model
2. **LSTM Network** (`lstm_model.py`) - Long Short-Term Memory for sequence modeling
3. **CNN Network** (`cnn_model.py`) - Convolutional Neural Network for text classification

## Data Preprocessing

### For Traditional ML Models:
- **Text Vectorization**: TF-IDF with max 5000 features, n-grams (1,2)
- **Stop Words**: English stop words removal
- **Label Encoding**: Convert -1/1 to 0/1 for binary classification

### For Deep Learning Models:
- **Tokenization**: Custom vocabulary building with padding
- **Sequence Length**: Max 100 tokens for LSTM/CNN, 128 for BERT
- **Special Tokens**: `<PAD>`, `<UNK>` for vocabulary handling

## Usage

### Training Individual Models

Run any model file to train and evaluate:

```bash
python logistic_regression_model.py
python bert_model.py
# etc.
```

### Model Comparison

Run the comparison script to evaluate all trained models:

```bash
python model_comparison.py
```

### Interactive Prediction

After training, each model provides an interactive interface for making predictions. Enter stock-related text to get sentiment analysis.

## Model Files

### Scikit-learn Models:
- `{model_name}_model.pkl`: Trained model
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer

### PyTorch Models:
- `{model_name}_model.pth`: Trained model weights
- `{model_name}_vocab.pkl`: Vocabulary for tokenization
- `./bert_tokenizer/`: BERT tokenizer files

## Performance Metrics

Models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
- **Precision/Recall/F1-Score**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification results

## Requirements

### Core Dependencies:
- Python 3.7+
- pandas
- numpy
- scikit-learn
- PyTorch
- joblib

### For BERT Model:
- transformers
- torch

## Installation

```bash
pip install pandas numpy scikit-learn torch transformers joblib
```

## Project Structure

```
Stock Sentiment Analysis/
├── stock_sentiment_analysis.py     # Main analysis file with EDA
├── logistic_regression_model.py    # Logistic Regression model
├── naive_bayes_model.py           # Naive Bayes model
├── svm_model.py                   # SVM model
├── random_forest_model.py         # Random Forest model
├── gradient_boosting_model.py     # Gradient Boosting model
├── bert_model.py                  # BERT model
├── lstm_model.py                  # LSTM model
├── cnn_model.py                   # CNN model
├── model_comparison.py            # Model comparison script
├── stock_data.csv                 # Dataset
├── README.md                      # This file
├── sentiment_analysis_visualization.png  # EDA plots
├── model_comparison_results.csv   # Comparison results
├── *_model.pkl                    # Trained sklearn models
├── *_model.pth                    # Trained PyTorch models
├── *.pkl                          # Saved preprocessors/vocabularies
└── bert_tokenizer/                # BERT tokenizer files
```

## Model Training Notes

- **Train/Test Split**: 80/20 stratified split
- **Random State**: 42 for reproducibility
- **BERT Training**: 3 epochs, batch size 16, AdamW optimizer
- **LSTM/CNN Training**: 10 epochs, batch size 32, Adam optimizer
- **Traditional ML**: Default hyperparameters with probability estimates

## Interactive Prediction Examples

```
Text: "AAPL stock is showing strong upward momentum today!"
Predicted sentiment: Positive (Confidence: 0.89)

Text: "Market crash expected, selling all positions"
Predicted sentiment: Negative (Confidence: 0.76)
```

## Best Practices

1. **Data Quality**: Ensure text data is clean and relevant to stock market
2. **Model Selection**: BERT typically performs best but requires more resources
3. **Hyperparameter Tuning**: Consider grid search for production use
4. **Real-time Deployment**: Use lighter models (Logistic Regression, SVM) for speed
5. **Model Updates**: Retrain periodically with new market data

## Future Enhancements

- **Multi-class Classification**: Bullish, Bearish, Neutral sentiments
- **Aspect-based Sentiment**: Analyze sentiment for specific stocks/companies
- **Time Series Integration**: Combine with price data for enhanced predictions
- **Ensemble Methods**: Combine multiple models for better performance
- **Domain Adaptation**: Fine-tune on financial news specifically