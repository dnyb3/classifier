"""
Diagnostic: trace why all snippets produce identical model outputs.
Run this from the same directory as inference_pipeline.py.

Checks:
  1. Model directory contents
  2. Tokenizer producing different tokens for different inputs
  3. Model loading warnings (missing/unexpected keys)
  4. Whether logits actually vary across inputs
  5. Whether the saved model matches what training produced
"""

import os
import sys
import warnings
import torch
import numpy as np

# Capture all warnings instead of printing to stderr
warning_log = []
original_warn = warnings.warn
def capture_warn(message, *args, **kwargs):
    warning_log.append(str(message))
    original_warn(message, *args, **kwargs)
warnings.warn = capture_warn

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from collections import OrderedDict

MODEL_PATH = "models/deberta_distribution/final_model/best_model"
MAX_LENGTH = 320

test_texts = [
    "i would recommend rolling this over to an ira because the fees are lower",
    "my suggestion is to stay in your current plan since it has great benefits",
    "i cant give you a recommendation on what to do with your account",
    "my recommendation would be to move this to your new employers plan at meridian",
]


def check_model_directory():
    print("=" * 60)
    print("1. MODEL DIRECTORY CONTENTS")
    print("=" * 60)
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: {MODEL_PATH} does not exist!")
        return False

    files = os.listdir(MODEL_PATH)
    print(f"  Path: {MODEL_PATH}")
    print(f"  Files ({len(files)}):")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(MODEL_PATH, f))
        print(f"    {f:40s}  {size:>12,} bytes")

    required = ['config.json']
    weight_files = [f for f in files if f.endswith('.safetensors') or f.endswith('.bin')]
    tokenizer_files = [f for f in files if 'spm' in f.lower() or 'tokenizer' in f.lower() or 'sentencepiece' in f.lower()]

    print(f"\n  Weight files: {weight_files or 'NONE FOUND'}")
    print(f"  Tokenizer files: {tokenizer_files or 'NONE FOUND'}")

    if not weight_files:
        print("  ERROR: No model weight files found!")
        return False
    if not tokenizer_files:
        print("  WARNING: No tokenizer files found - may fall back to hub")

    return True


def check_tokenizer():
    print(f"\n{'=' * 60}")
    print("2. TOKENIZER CHECK")
    print("=" * 60)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    except Exception as e:
        print(f"  ERROR loading tokenizer: {e}")
        print("  Falling back to hub tokenizer for comparison...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

    print(f"  Tokenizer class: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"  UNK token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")

    # Tokenize each test text separately
    print(f"\n  Tokenizing {len(test_texts)} test texts:")
    all_ids = []
    for i, text in enumerate(test_texts):
        enc = tokenizer(text, max_length=MAX_LENGTH, padding='max_length', truncation=True)
        ids = enc['input_ids']
        mask = enc['attention_mask']
        n_real = sum(mask)
        n_unk = ids.count(tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else 0
        all_ids.append(ids)
        print(f"    Text {i}: {n_real} real tokens, {n_unk} UNK tokens, first 10 ids: {ids[:10]}")

    # Check if all tokenizations are identical
    all_same = all(ids == all_ids[0] for ids in all_ids)
    print(f"\n  All tokenizations identical? {all_same}")
    if all_same:
        print("  *** THIS IS THE PROBLEM: tokenizer produces same tokens for all inputs ***")
        print("  The SPM model file is likely missing or corrupt.")
        return tokenizer, True
    else:
        print("  Tokenizer OK - producing different tokens for different inputs")

    return tokenizer, False


def check_model_loading():
    print(f"\n{'=' * 60}")
    print("3. MODEL LOADING CHECK")
    print("=" * 60)

    # Clear warning log
    warning_log.clear()

    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    print(f"  Config: hidden_size={config.hidden_size}, num_labels={config.num_labels}")
    print(f"  Model type: {config.model_type}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    model.eval()

    # Check for loading warnings
    load_warnings = [w for w in warning_log if 'weight' in w.lower() or 'key' in w.lower() or 'missing' in w.lower() or 'unexpected' in w.lower()]
    if load_warnings:
        print(f"\n  *** WEIGHT LOADING WARNINGS ({len(load_warnings)}): ***")
        for w in load_warnings[:10]:
            print(f"    {w[:200]}")
    else:
        print(f"  No weight loading warnings")

    # Check classifier head weights
    classifier_keys = [k for k in model.state_dict().keys() if 'classifier' in k or 'cls' in k]
    print(f"\n  Classifier head parameters:")
    for k in classifier_keys:
        v = model.state_dict()[k]
        print(f"    {k}: shape={list(v.shape)}, mean={v.mean():.6f}, std={v.std():.6f}, "
              f"min={v.min():.6f}, max={v.max():.6f}")

    # Check pooler weights
    pooler_keys = [k for k in model.state_dict().keys() if 'pooler' in k]
    print(f"\n  Pooler parameters:")
    for k in pooler_keys:
        v = model.state_dict()[k]
        print(f"    {k}: shape={list(v.shape)}, mean={v.mean():.6f}, std={v.std():.6f}")

    # Check a few encoder layer weights to verify they're not default init
    print(f"\n  Sample encoder weights (should NOT be near-zero mean with ~0.02 std if properly loaded):")
    for k in list(model.state_dict().keys())[:5]:
        v = model.state_dict()[k]
        print(f"    {k}: mean={v.float().mean():.6f}, std={v.float().std():.6f}")

    return model


def check_forward_pass(tokenizer, model):
    print(f"\n{'=' * 60}")
    print("4. FORWARD PASS CHECK")
    print("=" * 60)

    encodings = tokenizer(
        test_texts,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    print(f"  Input shape: {encodings['input_ids'].shape}")
    print(f"  Attention mask sum per row: {encodings['attention_mask'].sum(dim=1).tolist()}")

    # Check input_ids are different
    ids = encodings['input_ids']
    print(f"\n  Input IDs identical across batch? {(ids == ids[0]).all().item()}")
    for i in range(len(test_texts)):
        print(f"    Row {i} first 15 ids: {ids[i, :15].tolist()}")

    # Forward pass
    with torch.no_grad():
        outputs = model(**{k: v for k, v in encodings.items()})
        logits = outputs.logits

    print(f"\n  Raw logits:")
    for i in range(len(test_texts)):
        print(f"    Text {i}: [{', '.join(f'{v:.6f}' for v in logits[i].tolist())}]")

    # Check if logits are identical
    logits_np = logits.numpy()
    all_same = np.allclose(logits_np[0], logits_np[1:], atol=1e-6)
    print(f"\n  All logits identical? {all_same}")

    if all_same:
        print("  *** CONFIRMED: model produces identical outputs for different inputs ***")
        print()

        # Dig deeper - check intermediate representations
        print("  Checking intermediate representations...")
        base_model = model.deberta if hasattr(model, 'deberta') else model.base_model

        # Get encoder output
        with torch.no_grad():
            encoder_outputs = base_model(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
            )
            hidden = encoder_outputs.last_hidden_state  # (batch, seq, hidden)

        cls_tokens = hidden[:, 0, :]  # [CLS] representations
        cls_same = torch.allclose(cls_tokens[0], cls_tokens[1], atol=1e-4)
        print(f"  [CLS] representations identical? {cls_same}")
        print(f"  [CLS] row 0 first 10: {cls_tokens[0, :10].tolist()}")
        print(f"  [CLS] row 1 first 10: {cls_tokens[1, :10].tolist()}")

        if cls_same:
            # Check embeddings
            embed_layer = base_model.embeddings if hasattr(base_model, 'embeddings') else None
            if embed_layer:
                with torch.no_grad():
                    embeds = embed_layer(encodings['input_ids'])
                embed_same = torch.allclose(embeds[0], embeds[1], atol=1e-4)
                print(f"  Embeddings identical? {embed_same}")
                if embed_same:
                    print("  *** ROOT CAUSE: tokenizer producing identical tokens ***")
                else:
                    print("  Embeddings differ but encoder output is identical")
                    print("  *** ROOT CAUSE: encoder weights are broken (likely LayerNorm issue) ***")
        else:
            print("  [CLS] representations differ but logits are identical")
            print("  *** ROOT CAUSE: pooler or classifier head weights are broken ***")
    else:
        # Show probabilities
        probs = torch.softmax(logits, dim=-1)
        labels = ['no_advice', 'roll_ira', 'stay_plan', 'roll_new']
        print(f"\n  Probabilities:")
        for i in range(len(test_texts)):
            p = probs[i].tolist()
            pred = labels[np.argmax(p)]
            print(f"    Text {i}: {' | '.join(f'{l}={v:.3f}' for l, v in zip(labels, p))}  → {pred}")
        print(f"\n  Model is producing varied outputs - working correctly!")

    return logits


if __name__ == '__main__':
    print("INFERENCE PIPELINE DIAGNOSTIC")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Max length: {MAX_LENGTH}")
    print()

    # Step 1: Check files
    if not check_model_directory():
        sys.exit(1)

    # Step 2: Check tokenizer
    tokenizer, tokenizer_broken = check_tokenizer()
    if tokenizer_broken:
        print("\n\nFIX: Re-save the tokenizer from the training script or download fresh:")
        print("  from transformers import AutoTokenizer")
        print(f"  t = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')")
        print(f"  t.save_pretrained('{MODEL_PATH}')")
        sys.exit(1)

    # Step 3: Check model loading
    model = check_model_loading()

    # Step 4: Check forward pass
    logits = check_forward_pass(tokenizer, model)

    print(f"\n{'=' * 60}")
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
