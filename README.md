# S-A: Sentiment & Analysis ML Projects

This repository contains multiple machine-learning projects for classification and sentiment-analysis tasks across different domains.

## Repository Modules

- `Heart Disease Analysis/` — heart disease risk classification from tabular health data.
- `Imdb Analysis/` — movie review sentiment analysis.
- `Twitter Analysis/` — sentiment analysis workflows for Twitter-like text data.
- `Stock Sentiment Analysis/` — financial text sentiment classification.
- `Weather Analysis/` — weather-related classification experiments.
- `predictor.py` — simple interactive launcher for model scripts.

## Common Model Families

Across modules, the repository includes a mix of:

- Traditional ML models: Logistic Regression, Naive Bayes, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting
- Neural models: ANN, CNN, RNN, LSTM, GRU, BiLSTM
- Transformer-based model (in selected modules): BERT

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- nltk
- torch, transformers
- tensorflow (used in selected scripts)
- joblib

## Getting Started

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk torch transformers tensorflow joblib
```

### 3) (Optional) Download NLTK resources

Some sentiment scripts use NLTK corpora and tokenizers.

```bash
python -c "import nltk; nltk.download('movie_reviews'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## How to Run

Run scripts from inside the module directory so relative dataset paths resolve correctly.

### Example: Stock Sentiment Analysis

```bash
cd "Stock Sentiment Analysis"
python stock_sentiment_analysis.py
python logistic_regression_model.py
python model_comparison.py
```

### Example: Heart Disease Analysis

```bash
cd "Heart Disease Analysis"
python heart_disease_analysis.py
python random_forest_model.py
python model_comparison.py
```

### Other modules

```bash
cd "Imdb Analysis"
python sentiment_analysis.py

cd "../Twitter Analysis"
python sentiment_analysis.py

cd "../Weather Analysis"
python random_forest_model.py
python model_comparison.py
```

## Datasets

Datasets are stored within each module directory (for example `data.csv`, `stock_data.csv`, `Combined12.csv`, and IMDB/Twitter dataset files).  
If a script reports a missing file, confirm the dataset exists in that module folder and run the script from that folder.

## Notes

- Many model scripts are independent training/evaluation entry points.
- `model_comparison.py` scripts compare multiple trained models within a module.
- Training deep models may require more time and compute resources.

## License

This project is licensed under the terms in the `LICENSE` file.
