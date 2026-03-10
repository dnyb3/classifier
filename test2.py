import pandas as pd, numpy as np

# Load holdout exactly as orchestrator does
holdout_df = pd.read_csv("data/holdout.csv")
holdout_texts = holdout_df['passage'].tolist()
holdout_labels = holdout_df['type'].values.astype(int)
print(f"Holdout: {len(holdout_texts)} samples")
print(f"First text (first 80 chars): {holdout_texts[0][:80]}")
print(f"Any NaN: {pd.Series(holdout_texts).isna().sum()}")
print(f"Any empty: {sum(1 for t in holdout_texts if not t or t.strip() == '')}")

# Load DeBERTaScorer exactly as orchestrator does
from hybrid_inference_pipeline import DeBERTaScorer, HybridConfig
path = "models/deberta_distribution/final_model/best_model"
print(f"\nModel path: {path}")
config = HybridConfig(deberta_model_path=path)
print(f"ONNX enabled: {config.deberta_use_onnx}")

scorer = DeBERTaScorer(config)
print(f"ONNX session: {scorer.onnx_session is not None}")
print(f"PyTorch model: {scorer.model is not None}")

# Score first 10 through scorer
probs_10 = scorer.score_batch(holdout_texts[:10])
preds_10 = np.argmax(probs_10, axis=1)
print(f"\nFirst 10 labels: {holdout_labels[:10]}")
print(f"First 10 preds:  {preds_10}")
print(f"First 10 max p:  {probs_10.max(axis=1).round(3)}")
print(f"Accuracy (10):   {(preds_10 == holdout_labels[:10]).mean():.2f}")

# Score all through scorer (same loop as orchestrator)
all_probs = []
for start in range(0, len(holdout_texts), 64):
    batch = holdout_texts[start:start+64]
    all_probs.append(scorer.score_batch(batch))
probs = np.concatenate(all_probs, axis=0)
preds = np.argmax(probs, axis=1)
print(f"\nFull holdout accuracy: {(preds == holdout_labels).mean():.4f}")
print(f"Probs shape: {probs.shape}, Labels shape: {holdout_labels.shape}")
