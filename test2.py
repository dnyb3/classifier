print(f"Type A: {type(tok_a).__name__}")
print(f"Type B: {type(scorer.tokenizer).__name__}")

print(f"\nPath A resolved: {tok_a.name_or_path}")
print(f"Path B resolved: {scorer.tokenizer.name_or_path}")

# Check if spm.model is actually being loaded
print(f"\nVocab file A: {getattr(tok_a, 'vocab_file', 'NONE')}")
print(f"Vocab file B: {getattr(scorer.tokenizer, 'vocab_file', 'NONE')}")

# Check the tokenizer_config.json directly
import json, os
config_path = os.path.join(path, 'tokenizer_config.json')
if os.path.exists(config_path):
    with open(config_path) as f:
        tc = json.load(f)
    print(f"\ntokenizer_config.json:")
    print(f"  tokenizer_class: {tc.get('tokenizer_class', 'NOT SET')}")
    print(f"  model_type: {tc.get('model_type', 'NOT SET')}")
