# Stock Sentiment Analysis - Main Analysis File
# This script provides an overview and analysis of the stock sentiment dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_analyze_data():
    """Load and perform initial analysis of the stock sentiment dataset."""
    print("Loading Stock Sentiment Dataset...")
    df = pd.read_csv('stock_data.csv')

    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nClass Distribution:")
    print(df['Sentiment'].value_counts())
    print(f"Positive samples: {df['Sentiment'].value_counts()[1]} ({df['Sentiment'].value_counts()[1]/len(df)*100:.1f}%)")
    print(f"Negative samples: {df['Sentiment'].value_counts()[-1]} ({df['Sentiment'].value_counts()[-1]/len(df)*100:.1f}%)")

    # Basic text analysis
    print("\nText Analysis:")
    df['text_length'] = df['Text'].str.len()
    print(f"Average text length: {df['text_length'].mean():.1f} characters")
    print(f"Max text length: {df['text_length'].max()} characters")
    print(f"Min text length: {df['text_length'].min()} characters")

    # Sample texts
    print("\nSample Positive Texts:")
    positive_samples = df[df['Sentiment'] == 1]['Text'].head(3)
    for i, text in enumerate(positive_samples, 1):
        print(f"{i}. {text[:100]}...")

    print("\nSample Negative Texts:")
    negative_samples = df[df['Sentiment'] == -1]['Text'].head(3)
    for i, text in enumerate(negative_samples, 1):
        print(f"{i}. {text[:100]}...")

    return df

def visualize_data(df):
    """Create visualizations for the dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Sentiment distribution
    sentiment_counts = df['Sentiment'].value_counts()
    axes[0, 0].bar(['Positive (1)', 'Negative (-1)'], sentiment_counts.values,
                   color=['green', 'red'], alpha=0.7)
    axes[0, 0].set_title('Sentiment Distribution')
    axes[0, 0].set_ylabel('Count')

    # Text length distribution
    axes[0, 1].hist(df['text_length'], bins=50, alpha=0.7, color='blue')
    axes[0, 1].set_title('Text Length Distribution')
    axes[0, 1].set_xlabel('Text Length (characters)')
    axes[0, 1].set_ylabel('Frequency')

    # Text length by sentiment
    positive_lengths = df[df['Sentiment'] == 1]['text_length']
    negative_lengths = df[df['Sentiment'] == -1]['text_length']
    axes[1, 0].hist([positive_lengths, negative_lengths], bins=30, alpha=0.7,
                    label=['Positive', 'Negative'], color=['green', 'red'])
    axes[1, 0].set_title('Text Length by Sentiment')
    axes[1, 0].set_xlabel('Text Length (characters)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # Word count analysis
    df['word_count'] = df['Text'].str.split().str.len()
    positive_words = df[df['Sentiment'] == 1]['word_count']
    negative_words = df[df['Sentiment'] == -1]['word_count']
    axes[1, 1].hist([positive_words, negative_words], bins=20, alpha=0.7,
                    label=['Positive', 'Negative'], color=['green', 'red'])
    axes[1, 1].set_title('Word Count by Sentiment')
    axes[1, 1].set_xlabel('Word Count')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('sentiment_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Visualization saved as 'sentiment_analysis_visualization.png'")

def preprocess_data(df):
    """Preprocess the data for model training."""
    # Convert sentiment labels to 0 and 1 for sklearn compatibility
    df['Sentiment'] = df['Sentiment'].map({-1: 0, 1: 1})

    # Split the data
    X = df['Text']
    y = df['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nData Split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training class distribution: {y_train.value_counts().to_dict()}")
    print(f"Testing class distribution: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test

def main():
    """Main function for stock sentiment analysis."""
    print("=" * 60)
    print("STOCK SENTIMENT ANALYSIS")
    print("=" * 60)

    # Load and analyze data
    df = load_and_analyze_data()

    # Visualize data
    print("\n" + "=" * 40)
    print("DATA VISUALIZATION")
    print("=" * 40)
    visualize_data(df)

    # Preprocess data
    print("\n" + "=" * 40)
    print("DATA PREPROCESSING")
    print("=" * 40)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Available models to run:")
    print("1. Logistic Regression (logistic_regression_model.py)")
    print("2. Naive Bayes (naive_bayes_model.py)")
    print("3. SVM (svm_model.py)")
    print("4. Random Forest (random_forest_model.py)")
    print("5. Gradient Boosting (gradient_boosting_model.py)")
    print("6. BERT (bert_model.py)")
    print("7. LSTM (lstm_model.py)")
    print("8. CNN (cnn_model.py)")
    print("9. Model Comparison (model_comparison.py)")
    print("\nRun individual models or use model_comparison.py to compare all models.")

if __name__ == "__main__":
    main()