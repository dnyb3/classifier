import torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from hybrid_inference_pipeline import DeBERTaScorer, HybridConfig

path = "models/deberta_distribution/final_model/best_model"
holdout = pd.read_csv("data/holdout.csv")
texts = holdout['passage'].head(5).tolist()
labels = holdout['type'].head(5).values

# ── Path A: direct (works at 92%) ──
tok_a = AutoTokenizer.from_pretrained(path, local_files_only=True)
model_a = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
model_a.eval()
enc_a = tok_a(texts, padding=True, truncation=True, max_length=320, return_tensors='pt')
with torch.no_grad():
    logits_a = model_a(**enc_a).logits

# ── Path B: through DeBERTaScorer (gets 54%) ──
config = HybridConfig(deberta_model_path=path)
scorer = DeBERTaScorer(config)
enc_b = scorer.tokenizer(texts, padding=True, truncation=True,
                          max_length=config.deberta_max_length, return_tensors='pt')

with torch.no_grad():
    logits_b = scorer.model(**enc_b).logits

# ── Compare everything ──
print("Token IDs match:", torch.equal(enc_a['input_ids'], enc_b['input_ids']))
print("Attention mask match:", torch.equal(enc_a['attention_mask'], enc_b['attention_mask']))
print("Enc A keys:", list(enc_a.keys()))
print("Enc B keys:", list(enc_b.keys()))

print(f"\nLogits A (direct):\n{logits_a[:5].numpy().round(3)}")
print(f"Logits B (scorer):\n{logits_b[:5].numpy().round(3)}")
print(f"Logits match: {torch.allclose(logits_a, logits_b, atol=1e-4)}")

print(f"\nPreds A: {logits_a.argmax(dim=-1).numpy()}")
print(f"Preds B: {logits_b.argmax(dim=-1).numpy()}")
print(f"Labels:  {labels}")

# Check if models are the same object in memory
print(f"\nModel A params: {sum(p.sum().item() for p in model_a.parameters()):.4f}")
print(f"Model B params: {sum(p.sum().item() for p in scorer.model.parameters()):.4f}")
