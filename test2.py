print(f"A is_fast: {tok_a.is_fast}")
print(f"B is_fast: {scorer.tokenizer.is_fast}")

# Check what files exist
import os
model_dir = "models/deberta_distribution/final_model/best_model"
for f in ['tokenizer.json', 'spm.model', 'tokenizer_config.json']:
    full = os.path.join(model_dir, f)
    exists = os.path.exists(full)
    size = os.path.getsize(full) if exists else 0
    print(f"  {f}: {'exists' if exists else 'MISSING'} ({size/1024:.0f} KB)")
