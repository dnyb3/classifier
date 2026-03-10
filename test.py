import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

path = "models/deberta_distribution/final_model/best_model"
holdout = pd.read_csv("data/holdout.csv")

tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
model.eval()

# Score first 20 holdout samples
texts = holdout['passage'].head(20).tolist()
labels = holdout['type'].head(20).values

enc = tok(texts, padding=True, truncation=True, max_length=320, return_tensors='pt')
with torch.no_grad():
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).numpy()

preds = np.argmax(probs, axis=1)
print("Labels:", labels)
print("Preds: ", preds)
print("Probs max:", probs.max(axis=1).round(3))
print("Accuracy:", (preds == labels).mean())

# Check if probs are near-uniform (the smoking gun)
print("\nFirst 5 prob rows:")
print(probs[:5].round(3))
