# Airline Sentiment Analysis - Main Analysis File
# This script provides an overview and analysis of the airline sentiment dataset.

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
    """Load and perform initial analysis of the airline sentiment dataset."""
    print("Loading Airline Sentiment Dataset...")
    df = pd.read_csv('Tweets.csv')

    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nSentiment Distribution:")
    print(df['airline_sentiment'].value_counts())
    print(f"Negative samples: {df['airline_sentiment'].value_counts()['negative']} ({df['airline_sentiment'].value_counts()['negative']/len(df)*100:.1f}%)")
    print(f"Neutral samples: {df['airline_sentiment'].value_counts()['neutral']} ({df['airline_sentiment'].value_counts()['neutral']/len(df)*100:.1f}%)")
    print(f"Positive samples: {df['airline_sentiment'].value_counts()['positive']} ({df['airline_sentiment'].value_counts()['positive']/len(df)*100:.1f}%)")

    # Basic text analysis
    print("\nText Analysis:")
    df['text_length'] = df['text'].str.len()
    print(f"Average text length: {df['text_length'].mean():.1f} characters")
    print(f"Max text length: {df['text_length'].max()} characters")
    print(f"Min text length: {df['text_length'].min()} characters")

    # Airline distribution
    print("\nAirline Distribution:")
    print(df['airline'].value_counts())

    # Sample texts by sentiment
    print("\nSample Negative Texts:")
    negative_samples = df[df['airline_sentiment'] == 'negative']['text'].head(2)
    for i, text in enumerate(negative_samples, 1):
        print(f"{i}. {text[:100]}...")

    print("\nSample Neutral Texts:")
    neutral_samples = df[df['airline_sentiment'] == 'neutral']['text'].head(2)
    for i, text in enumerate(neutral_samples, 1):
        print(f"{i}. {text[:100]}...")

    print("\nSample Positive Texts:")
    positive_samples = df[df['airline_sentiment'] == 'positive']['text'].head(2)
    for i, text in enumerate(positive_samples, 1):
        print(f"{i}. {text[:100]}...")

    return df

def visualize_data(df):
    """Create visualizations for the dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Sentiment distribution
    sentiment_counts = df['airline_sentiment'].value_counts()
    axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values,
                   color=['red', 'orange', 'green'], alpha=0.7)
    axes[0, 0].set_title('Sentiment Distribution')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Airline distribution
    airline_counts = df['airline'].value_counts()
    axes[0, 1].bar(airline_counts.index, airline_counts.values, alpha=0.7)
    axes[0, 1].set_title('Tweets by Airline')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Text length distribution
    axes[0, 2].hist(df['text_length'], bins=50, alpha=0.7, color='blue')
    axes[0, 2].set_title('Text Length Distribution')
    axes[0, 2].set_xlabel('Text Length (characters)')
    axes[0, 2].set_ylabel('Frequency')

    # Text length by sentiment
    negative_lengths = df[df['airline_sentiment'] == 'negative']['text_length']
    neutral_lengths = df[df['airline_sentiment'] == 'neutral']['text_length']
    positive_lengths = df[df['airline_sentiment'] == 'positive']['text_length']
    axes[1, 0].hist([negative_lengths, neutral_lengths, positive_lengths], bins=30, alpha=0.7,
                    label=['Negative', 'Neutral', 'Positive'], color=['red', 'orange', 'green'])
    axes[1, 0].set_title('Text Length by Sentiment')
    axes[1, 0].set_xlabel('Text Length (characters)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # Sentiment by airline
    sentiment_airline = pd.crosstab(df['airline'], df['airline_sentiment'])
    sentiment_airline.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
    axes[1, 1].set_title('Sentiment Distribution by Airline')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='Sentiment')

    # Word count analysis
    df['word_count'] = df['text'].str.split().str.len()
    negative_words = df[df['airline_sentiment'] == 'negative']['word_count']
    neutral_words = df[df['airline_sentiment'] == 'neutral']['word_count']
    positive_words = df[df['airline_sentiment'] == 'positive']['word_count']
    axes[1, 2].hist([negative_words, neutral_words, positive_words], bins=20, alpha=0.7,
                    label=['Negative', 'Neutral', 'Positive'], color=['red', 'orange', 'green'])
    axes[1, 2].set_title('Word Count by Sentiment')
    axes[1, 2].set_xlabel('Word Count')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig('airline_sentiment_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Visualization saved as 'airline_sentiment_analysis_visualization.png'")

def preprocess_data(df):
    """Preprocess the data for model training."""
    # Map sentiment labels to numerical values
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_label'] = df['airline_sentiment'].map(sentiment_mapping)

    # Select relevant columns
    df_processed = df[['text', 'sentiment_label', 'airline_sentiment']].copy()

    # Split the data
    X = df_processed['text']
    y = df_processed['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nData Split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training class distribution: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Testing class distribution: {y_test.value_counts().sort_index().to_dict()}")

    return X_train, X_test, y_train, y_test, df_processed

def main():
    """Main function for airline sentiment analysis."""
    print("=" * 60)
    print("AIRLINE SENTIMENT ANALYSIS")
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
    X_train, X_test, y_train, y_test, df_processed = preprocess_data(df)

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