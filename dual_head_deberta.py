"""
Dual-Head DeBERTa: Binary + Multiclass from One Encoder
========================================================
One DeBERTa forward pass produces two outputs:
  - binary_logits:  (batch, 2) — advice vs no-advice (for ranking)
  - multi_logits:   (batch, 4) — advice type (for reviewer context)

The encoder + pooler are shared. Only the final linear projections differ.
Cost: ~0.01% more parameters and microseconds per batch vs single-head.

Training:
  Combined loss = binary_loss + multiclass_loss (equal weight by default).
  Both heads receive gradient through the shared encoder, so the encoder
  learns features useful for BOTH tasks simultaneously.

Saving/Loading:
  Saves as a directory with:
    - model_state.pt    (full state dict)
    - model_config.json (architecture config)
    - tokenizer files   (copied from pretrained)
  Loads by reconstructing the architecture from config and loading state dict.
  No dependency on HuggingFace's save_pretrained/from_pretrained.
"""

import os
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    accuracy_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)

from augment_training_data import run_augmentation_pipeline


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

class DualHeadDeBERTa(nn.Module):
    """
    DeBERTa encoder + pooler shared between two classification heads.

    Forward pass:
      input_ids, attention_mask
        → DeBERTa encoder → sequence_output
        → ContextPooler (from DeBERTa) or mean-pool → pooled
        → dropout
        → head_binary(pooled) → binary_logits  (batch, 2)
        → head_multi(pooled)  → multi_logits   (batch, num_labels)
    """

    def __init__(self, encoder, pooler, hidden_size, num_labels=4, dropout_rate=0.1,
                 loss_fn_multi=None, loss_fn_binary=None):
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.dropout = nn.Dropout(dropout_rate)
        self.head_binary = nn.Linear(hidden_size, 2)
        self.head_multi = nn.Linear(hidden_size, num_labels)
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        # Store loss functions in a plain dict to avoid nn.Module __setattr__
        # registering FocalLoss as a submodule (which would pollute the state_dict
        # and break loading from disk when loss config differs).
        self._loss_fns = {
            'multi': loss_fn_multi or F.cross_entropy,
            'binary': loss_fn_binary or F.cross_entropy,
        }

    def forward(self, input_ids, attention_mask=None, labels=None, labels_binary=None, **kwargs):
        # Encoder forward
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = encoder_output.last_hidden_state  # (batch, seq_len, hidden)

        # Pool — use DeBERTa's ContextPooler if available, else use [CLS]
        if self.pooler is not None:
            pooled = self.pooler(sequence_output)
        else:
            pooled = sequence_output[:, 0, :]  # [CLS] token

        pooled = self.dropout(pooled)

        # Two heads, one pooled representation
        multi_logits = self.head_multi(pooled)
        binary_logits = self.head_binary(pooled)

        # Compute loss when labels provided (training and evaluation)
        loss = None
        if labels is not None and labels_binary is not None:
            loss_multi = self._loss_fns['multi'](multi_logits, labels)
            loss_binary = self._loss_fns['binary'](binary_logits, labels_binary)
            loss = loss_multi + loss_binary

        return DualHeadOutput(
            loss=loss,
            multi_logits=multi_logits,
            binary_logits=binary_logits,
        )


class DualHeadOutput:
    """Output container compatible with HuggingFace Trainer.
    
    Supports attribute access (.loss, .logits) and tuple-style indexing
    (outputs[0] = loss, outputs[1:] = (logits,)) for Trainer compatibility.
    """
    def __init__(self, loss, multi_logits, binary_logits):
        self.loss = loss
        self.logits = torch.cat([multi_logits, binary_logits], dim=-1)
        self.multi_logits = multi_logits
        self.binary_logits = binary_logits
        self._tuple = (loss, self.logits)

    def __getitem__(self, idx):
        return self._tuple[idx]

    def __iter__(self):
        return iter(self._tuple)

    def __len__(self):
        return len(self._tuple)


def load_dual_head_model(model_name: str, num_labels: int = 4, dropout_rate: float = 0.1,
                         loss_fn_multi=None, loss_fn_binary=None):
    """
    Load pretrained DeBERTa encoder and wrap with dual classification heads.
    Handles the DeBERTa-v3 LayerNorm beta/gamma remapping.
    """
    print(f"  Loading DualHeadDeBERTa from {model_name}")

    # Load a full classification model to get encoder + pooler with correct weights
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    ref_model = AutoModelForSequenceClassification.from_config(config)

    pretrained = AutoModel.from_pretrained(model_name)
    pretrained_state = pretrained.state_dict()

    # Remap beta -> bias, gamma -> weight (DeBERTa-v3 quirk)
    remapped = OrderedDict()
    for key, value in pretrained_state.items():
        new_key = key
        if '.LayerNorm.beta' in key:
            new_key = key.replace('.LayerNorm.beta', '.LayerNorm.bias')
        elif '.LayerNorm.gamma' in key:
            new_key = key.replace('.LayerNorm.gamma', '.LayerNorm.weight')
        remapped[new_key] = value

    # Load into ref_model to get properly initialized encoder + pooler
    ref_state = ref_model.state_dict()
    loaded = 0
    for key, value in remapped.items():
        if key in ref_state and ref_state[key].shape == value.shape:
            ref_state[key] = value
            loaded += 1
        elif f"deberta.{key}" in ref_state and ref_state[f"deberta.{key}"].shape == value.shape:
            ref_state[f"deberta.{key}"] = value
            loaded += 1
    ref_model.load_state_dict(ref_state)

    ln_loaded = sum(1 for k in list(remapped.keys()) if 'LayerNorm' in k)
    print(f"  Pretrained params mapped: {loaded}/{len(remapped)}")

    # Extract encoder and pooler
    encoder = ref_model.deberta
    pooler = ref_model.pooler if hasattr(ref_model, 'pooler') else None

    # Use pooler's output dimension for heads, not encoder's hidden_size.
    # DeBERTa's ContextPooler can output a different size than the encoder.
    if pooler is not None and hasattr(pooler, 'output_dim'):
        head_input_size = pooler.output_dim
    else:
        head_input_size = config.hidden_size
    print(f"  Encoder hidden: {config.hidden_size}, Head input: {head_input_size}, "
          f"Pooler: {'yes' if pooler else 'no'}")

    # Build dual-head model
    model = DualHeadDeBERTa(
        encoder=encoder,
        pooler=pooler,
        hidden_size=head_input_size,
        num_labels=num_labels,
        dropout_rate=dropout_rate,
        loss_fn_multi=loss_fn_multi,
        loss_fn_binary=loss_fn_binary,
    )

    # Clean up
    del pretrained, pretrained_state, remapped, ref_model
    gc.collect()

    total_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in model.head_binary.parameters()) + \
                  sum(p.numel() for p in model.head_multi.parameters())
    print(f"  Total params: {total_params:,} (heads: {head_params:,}, "
          f"{100*head_params/total_params:.2f}%)")

    return model


def save_dual_head_model(model, tokenizer, save_dir, base_model_name="microsoft/deberta-v3-small"):
    """Save model state dict, config, and tokenizer."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model_state.pt"))

    # Save architecture config (includes base_model_name for reconstruction)
    config = {
        'hidden_size': model.hidden_size,
        'num_labels': model.num_labels,
        'model_class': 'DualHeadDeBERTa',
        'base_model_name': base_model_name,
    }
    with open(os.path.join(save_dir, "model_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    # Tokenizer smoke test
    try:
        test_tok = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
        t1 = test_tok.encode("i recommend rolling over to an ira")
        t2 = test_tok.encode("the weather is nice today")
        if t1[:5] == t2[:5]:
            print(f"  WARNING: Saved tokenizer produces identical tokens — "
                  f"inference will be broken!")
        else:
            print(f"  Tokenizer smoke test passed")
        del test_tok
    except Exception as e:
        print(f"  Tokenizer smoke test failed: {e}")

    print(f"  Dual-head model saved to {save_dir}")


def load_dual_head_from_disk(save_dir, base_model_name=None):
    """
    Load a saved dual-head model from disk.

    Reconstructs the architecture from the saved config and loads the state dict.
    Does NOT download pretrained weights — the state dict contains everything.
    Still needs the base_model_name to reconstruct the encoder architecture
    (config shapes, attention patterns, etc.) but uses from_config not from_pretrained.
    """
    config_path = os.path.join(save_dir, "model_config.json")
    state_path = os.path.join(save_dir, "model_state.pt")

    with open(config_path, 'r') as f:
        saved_config = json.load(f)

    # Use saved base_model_name if available, fall back to parameter
    model_name = saved_config.get('base_model_name', base_model_name or 'microsoft/deberta-v3-small')

    # Build architecture from config only (no pretrained weight download)
    print(f"  Rebuilding architecture from {model_name} config...")
    hf_config = AutoConfig.from_pretrained(model_name, num_labels=saved_config['num_labels'])
    ref_model = AutoModelForSequenceClassification.from_config(hf_config)

    encoder = ref_model.deberta
    pooler = ref_model.pooler if hasattr(ref_model, 'pooler') else None

    if pooler is not None and hasattr(pooler, 'output_dim'):
        head_input_size = pooler.output_dim
    else:
        head_input_size = hf_config.hidden_size

    model = DualHeadDeBERTa(
        encoder=encoder,
        pooler=pooler,
        hidden_size=head_input_size,
        num_labels=saved_config['num_labels'],
    )

    del ref_model
    gc.collect()

    # Load saved weights (overwrites all random initialization)
    state_dict = torch.load(state_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  Loaded dual-head model from {save_dir}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class DualLabelDataset(Dataset):
    """Dataset that provides both 4-class and binary labels."""

    def __init__(self, encodings, labels_multi):
        self.encodings = encodings
        self.labels_multi = labels_multi
        # Binary: 0 = no advice, 1 = any advice (classes 1,2,3 collapsed)
        self.labels_binary = (labels_multi > 0).astype(int)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels_multi[idx]).long()
        item['labels_binary'] = torch.tensor(self.labels_binary[idx]).long()
        return item

    def __len__(self):
        return len(self.labels_multi)


# ═══════════════════════════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced classification.
    alpha: uniform loss scaling (1.0 = no scaling, 0.25 = original RetinaNet)
    gamma: difficulty focusing (0 = CE, 0.5 = mild, 2.0 = aggressive)
    """
    def __init__(self, alpha=1.0, gamma=0.5, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')

        probs = log_probs.exp().clamp(self.eps, 1.0 - self.eps)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma

        loss = focal_weight * ce_loss
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════

class DualHeadTrainer(Trainer):
    """
    Trainer for dual-head model. Loss is computed in model.forward().
    We override label_names so the Trainer collects both label columns
    during evaluation (otherwise labels_binary gets dropped).
    """

    @property
    def label_names(self):
        return ["labels", "labels_binary"]

    @label_names.setter
    def label_names(self, value):
        # Trainer.__init__ tries to set this; ignore it and keep ours
        pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def encode_data(texts, tokenizer, max_length=320):
    return tokenizer(
        texts, max_length=max_length, padding='max_length',
        truncation=True, return_tensors=None
    )


def augment_and_balance(df, label_col='type', max_class0_ratio=0.45):
    augmented = run_augmentation_pipeline(df, passage_col='passage', label_col=label_col)
    class0 = augmented[augmented[label_col] == 0]
    other = augmented[augmented[label_col] != 0]
    target = int(len(other) * max_class0_ratio / (1.0 - max_class0_ratio))
    if len(class0) > target and 'source' in class0.columns:
        orig = class0[class0['source'] == 'original']
        aug = class0[class0['source'] != 'original']
        n_keep = max(target - len(orig), 0)
        if n_keep < len(aug):
            aug = aug.sample(n=n_keep, random_state=42)
        class0 = pd.concat([orig, aug])
    elif len(class0) > target:
        class0 = class0.sample(n=target, random_state=42)
    result = pd.concat([class0, other]).reset_index(drop=True)
    if 'source' in result.columns:
        result = result.drop(columns=['source'])
    return result


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dual_head(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    tokenizer,
    config: dict,
    output_dir: str,
) -> Tuple[str, Optional[Dict], Optional[Tuple]]:
    """
    Train a single dual-head model.

    Args:
        config: dict with keys like num_labels, max_length, learning_rate, etc.

    Returns:
        (model_path, eval_metrics, (multi_logits, binary_logits, labels))
    """
    os.makedirs(output_dir, exist_ok=True)
    model_name = config.get('model_name', 'microsoft/deberta-v3-small')

    # Encode
    train_enc = encode_data(train_df['passage'].tolist(), tokenizer, config.get('max_length', 320))
    train_labels = train_df['type'].values.astype(int)
    train_dataset = DualLabelDataset(train_enc, train_labels)

    val_dataset = None
    val_labels = None
    if val_df is not None:
        val_enc = encode_data(val_df['passage'].tolist(), tokenizer, config.get('max_length', 320))
        val_labels = val_df['type'].values.astype(int)
        val_dataset = DualLabelDataset(val_enc, val_labels)

    has_val = val_dataset is not None

    # Loss functions — focal loss or plain CE
    loss_fn_multi = None
    loss_fn_binary = None

    if config.get('use_focal_loss', True):
        focal_alpha = config.get('focal_alpha', 1.0)
        focal_gamma = config.get('focal_gamma', 0.5)
        loss_fn_multi = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        loss_fn_binary = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"  Loss: Focal (alpha={focal_alpha}, gamma={focal_gamma})")
    else:
        print(f"  Loss: CrossEntropy")

    # Load model with loss functions attached
    model = load_dual_head_model(
        model_name,
        num_labels=config.get('num_labels', 4),
        dropout_rate=config.get('dropout_rate', 0.1),
        loss_fn_multi=loss_fn_multi,
        loss_fn_binary=loss_fn_binary,
    )

    # Metrics — logits are concatenated [multi(4), binary(2)]
    num_labels = config.get('num_labels', 4)

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred

        # Trainer collects both 'labels' and 'labels_binary' into a tuple
        if isinstance(label_ids, tuple):
            labels_multi, binary_true = label_ids
        else:
            labels_multi = label_ids
            binary_true = (labels_multi > 0).astype(int)

        # Split concatenated logits [multi(4), binary(2)]
        multi_logits = logits[:, :num_labels]
        binary_logits = logits[:, num_labels:]

        multi_preds = np.argmax(multi_logits, axis=-1)
        binary_preds = np.argmax(binary_logits, axis=-1)

        return {
            'f1_macro': f1_score(labels_multi, multi_preds, average='macro', zero_division=0),
            'accuracy': accuracy_score(labels_multi, multi_preds),
            'precision_macro': precision_score(labels_multi, multi_preds, average='macro', zero_division=0),
            'binary_f1': f1_score(binary_true, binary_preds, average='binary', zero_division=0),
            'binary_accuracy': accuracy_score(binary_true, binary_preds),
            'binary_precision': precision_score(binary_true, binary_preds, zero_division=0),
            'binary_recall': recall_score(binary_true, binary_preds, zero_division=0),
        }

    # Training args
    steps_per_epoch = max(
        len(train_dataset) // config.get('batch_size', 16), 1
    )
    total_steps = steps_per_epoch * config.get('num_epochs', 15)
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy='epoch' if has_val else 'no',
        save_strategy='epoch' if has_val else 'no',
        learning_rate=config.get('learning_rate', 3e-5),
        num_train_epochs=config.get('num_epochs', 15),
        per_device_train_batch_size=config.get('batch_size', 16),
        per_device_eval_batch_size=config.get('eval_batch_size', 32),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        weight_decay=config.get('weight_decay', 0.01),
        warmup_steps=warmup_steps,
        logging_steps=10,
        load_best_model_at_end=has_val,
        metric_for_best_model='eval_f1_macro' if has_val else None,
        greater_is_better=True if has_val else None,
        save_total_limit=1,
        seed=config.get('seed', 42),
        fp16=(get_device() == "cuda"),
        gradient_checkpointing=config.get('gradient_checkpointing', False),
        remove_unused_columns=False,  # keep labels_binary
    )

    callbacks = []
    if has_val:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config.get('early_stopping_patience', 5),
            early_stopping_threshold=0.001,
        ))

    trainer = DualHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics if has_val else None,
        callbacks=callbacks,
    )

    trainer.train()

    # Log best checkpoint
    if has_val and hasattr(trainer.state, 'best_model_checkpoint'):
        print(f"\n  Best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"  Best metric: {trainer.state.best_metric:.4f}")

    # Evaluate
    eval_metrics = None
    val_logits_tuple = None
    if has_val:
        eval_metrics = trainer.evaluate()
        print(f"  Results: {eval_metrics}")

        predictions = trainer.predict(val_dataset)
        all_logits = predictions.predictions
        multi_logits = all_logits[:, :num_labels]
        binary_logits = all_logits[:, num_labels:]

        multi_preds = np.argmax(multi_logits, axis=-1)
        binary_true = (val_labels > 0).astype(int)
        binary_preds = np.argmax(binary_logits, axis=-1)

        print(f"\n  Multiclass Report:")
        print(classification_report(val_labels, multi_preds, zero_division=0))
        print(f"  Binary Report (advice vs no-advice):")
        print(classification_report(binary_true, binary_preds,
                                    target_names=['no_advice', 'advice'], zero_division=0))

        val_logits_tuple = (multi_logits, binary_logits, val_labels)

    # Save
    model_path = os.path.join(output_dir, "best_model")
    save_dual_head_model(model, tokenizer, model_path, base_model_name=model_name)

    # Free memory
    del trainer, model
    gc.collect()

    return model_path, eval_metrics, val_logits_tuple


# ═══════════════════════════════════════════════════════════════════════════
# K-Fold + Final Training
# ═══════════════════════════════════════════════════════════════════════════

def train_dual_head_pipeline(
    data_path: str,
    output_dir: str,
    model_name: str = "microsoft/deberta-v3-small",
    seed: int = 42,
    n_folds: int = 5,
    **config_overrides,
):
    """
    Full dual-head training pipeline:
      1. K-fold validation (reports both binary and multiclass metrics)
      2. Train final model on all data
    """
    print("=" * 60)
    print("DUAL-HEAD DEBERTA TRAINING")
    print("=" * 60)

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Config
    config = {
        'model_name': model_name,
        'num_labels': 4,
        'max_length': 320,
        'learning_rate': 3e-5,
        'num_epochs': 15,
        'batch_size': 16,
        'eval_batch_size': 32,
        'gradient_accumulation_steps': 1,
        'gradient_checkpointing': False,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'dropout_rate': 0.1,
        'early_stopping_patience': 5,
        'seed': seed,
        'max_class0_ratio': 0.45,
        # Focal loss — mild difficulty focusing, no dampening
        'use_focal_loss': True,
        'focal_alpha': 1.0,    # no uniform dampening
        'focal_gamma': 0.5,    # mild focus on hard examples
    }
    config.update(config_overrides)

    # Save config
    with open(os.path.join(output_dir, "train_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Load data
    df = pd.read_csv(data_path)
    df['passage'] = df['passage'].str.lower().str.strip()
    df['passage'] = df['passage'].str.replace(r"[^\w\s']", '', regex=True)
    df = df.dropna(subset=['type', 'passage'])
    df['type'] = df['type'].astype(int)
    print(f"\nLoaded {len(df)} samples")
    print(f"Labels: {df['type'].value_counts().sort_index().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── K-fold ──
    print(f"\n{'='*60}")
    print(f"K-FOLD VALIDATION ({n_folds} folds)")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['type'])):
        print(f"\n{'_'*40}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'_'*40}")

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        metrics_path = os.path.join(fold_dir, "eval_metrics.json")

        # Resume
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print(f"  Fold complete — f1={metrics.get('eval_f1_macro', 0):.3f}, "
                  f"binary_f1={metrics.get('eval_binary_f1', 0):.3f}")
            fold_results.append(metrics)
            continue

        set_seed(seed + fold)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Augment train only
        aug_train = augment_and_balance(train_df, max_class0_ratio=config['max_class0_ratio'])
        print(f"  Train: {len(train_df)} -> {len(aug_train)} | Val: {len(val_df)}")

        _, metrics, _ = train_dual_head(aug_train, val_df, tokenizer, config, fold_dir)

        if metrics:
            fold_results.append(metrics)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

        gc.collect()

    # K-fold summary
    print(f"\n{'='*60}")
    print("K-FOLD SUMMARY")
    print("=" * 60)
    for key in ['eval_f1_macro', 'eval_accuracy', 'eval_binary_f1',
                'eval_binary_precision', 'eval_binary_recall']:
        vals = [r.get(key, 0) for r in fold_results if key in r]
        if vals:
            print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # ── Final model ──
    print(f"\n{'='*60}")
    print("FINAL MODEL (all data)")
    print("=" * 60)

    final_dir = os.path.join(output_dir, "final_model")
    final_model_path = os.path.join(final_dir, "best_model")

    if os.path.exists(os.path.join(final_model_path, "model_state.pt")):
        print("  Final model already trained, skipping")
    else:
        set_seed(seed)
        aug_all = augment_and_balance(df, max_class0_ratio=config['max_class0_ratio'])
        print(f"  Full: {len(df)} -> {len(aug_all)} (augmented)")

        # Small monitor split for loss tracking
        from sklearn.model_selection import train_test_split
        train_split, monitor_split = train_test_split(
            aug_all, test_size=0.05, stratify=aug_all['type'], random_state=seed
        )

        train_dual_head(train_split, monitor_split, tokenizer, config, final_dir)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"  Model: {final_model_path}")
    print("=" * 60)

    return final_model_path


# ═══════════════════════════════════════════════════════════════════════════
# Scorer (for use in hybrid pipeline)
# ═══════════════════════════════════════════════════════════════════════════

class DualHeadScorer:
    """
    Loads a saved dual-head model and scores text snippets.

    Returns both binary and multiclass probabilities.
    Binary probs are used for ranking (is this advice at all?).
    Multiclass probs tell reviewers which type of advice.
    """

    def __init__(self, model_dir, max_length=320, batch_size=64):
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = load_dual_head_from_disk(model_dir)
        self.model.eval()

        self.device = get_device()
        if self.device != 'cpu':
            self.model.to(self.device)

    def score_batch(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score a batch of texts.

        Returns:
            multi_probs:  (n, num_labels) — 4-class probabilities
            binary_probs: (n, 2) — [no_advice_prob, advice_prob]
        """
        if not texts:
            return np.array([]), np.array([])

        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        if self.device != 'cpu':
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            multi_probs = F.softmax(outputs.multi_logits, dim=-1).cpu().numpy()
            binary_probs = F.softmax(outputs.binary_logits, dim=-1).cpu().numpy()

        return multi_probs, binary_probs

    def score_all(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Score all texts in batches."""
        all_multi = []
        all_binary = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            multi, binary = self.score_batch(batch)
            all_multi.append(multi)
            all_binary.append(binary)

        return np.concatenate(all_multi), np.concatenate(all_binary)
