# Model Comparison Script for Weather Data
# This script runs all weather classification models and compares their accuracies.

import subprocess
import sys

models = [
    'naive_bayes_model.py',
    'svm_model.py',
    'random_forest_model.py',
    'logistic_regression_model.py',
    'knn_model.py',
    'decision_tree_model.py',
    'ann_model.py',
    'cnn_model.py',
    'rnn_model.py',
    'lstm_model.py',
    'gru_model.py',
    'bilstm_model.py',
]

results = {}

for model in models:
    print(f"\nRunning {model}...")
    try:
        result = subprocess.run([sys.executable, model], capture_output=True, text=True, cwd=r'e:\S A\Weather Analysis')
        output = result.stdout
        # Extract accuracy from output
        for line in output.split('\n'):
            if 'Accuracy:' in line:
                acc = float(line.split(':')[1].strip())
                results[model] = acc
                print(f"{model}: {acc:.2f}")
                break
    except Exception as e:
        print(f"Error running {model}: {e}")

print("\nModel Comparison:")
for model, acc in results.items():
    print(f"{model}: {acc:.2f}")

best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} with accuracy {results[best_model]:.2f}")