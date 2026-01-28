import os
import subprocess
import sys

datasets = ['Airline Sentiment Analysis', 'Delhi Trafiic Analysis', 'Heart Disease Analysis', 'Imdb Analysis', 'Stock Sentiment Analysis', 'Twitter Analysis', 'Weather Analysis']

print("Select dataset:")
for i, d in enumerate(datasets, 1):
    print(f"{i}. {d}")

dataset_choice = int(input("Enter number: ")) - 1
dataset = datasets[dataset_choice]

os.chdir(dataset)

models = [f for f in os.listdir('.') if f.endswith('_model.py') and f != 'model_comparison.py']

print("Select model:")
for i, m in enumerate(models, 1):
    print(f"{i}. {m}")

model_choice = int(input("Enter number: ")) - 1
model = models[model_choice]

print(f"Running {model} in {dataset}...")
print("The model will train, show evaluation metrics, and then allow prediction on user input.")
print("Press enter to skip prediction if desired.")

subprocess.run([sys.executable, model])