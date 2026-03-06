"""
Stage 2: DeBERTa-v3-small Training for Distribution Advice Detection
=====================================================================

Workflow:
  1. Load raw training data (original labeled passages)
  2. K-fold cross-validation to validate architecture + hyperparameters
     - Augmentation happens INSIDE each fold on the training split only
     - This prevents data leakage from augmented variants
  3. Train final production model on 100% of the data
  4. Save final model for inference
"""

import gc
import random
import os
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
    classification_report, f1_score, precision_score, recall_score
)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    AutoModel,
)

from augment_training_data import run_augmentation_pipeline
from calibration import TemperatureScaler, expected_calibration_error


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model_name: str = "microsoft/deberta-v3-small"
    num_labels: int = 4
    max_length: int = 320  # Covers 100% of training passages + inference snippets (60+150 word window ≈ 275 tokens)

    # Training — memory-optimized for Apple Silicon / modest GPU
    learning_rate: float = 2e-5
    num_epochs: int = 15
    batch_size: int = 8
    eval_batch_size: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4   # Effective batch = 8 * 4 = 32
    gradient_checkpointing: bool = True    # Trades ~20% speed for ~40% less memory

    # Focal loss
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Label smoothing
    label_smoothing: float = 0.05

    # K-fold (for validation only)
    n_folds: int = 5

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001

    # Augmentation
    max_class0_ratio: float = 0.45  # Cap class 0 at 45% of augmented data

    # Paths
    data_path: str = "training_data/model_cases7.csv"
    output_dir: str = "models/deberta_distribution"

    # Reproducibility
    seed: int = 42

    # Label names
    label_names: Dict[int, str] = field(default_factory=lambda: {
        0: "no_advice",
        1: "roll_to_ira",
        2: "stay_in_plan",
        3: "roll_to_other_plan",
    })


# ── DeBERTa-v3 LayerNorm Key Remapping ───────────────────────────────────────

def load_deberta_v3_model(model_name: str, num_labels: int):
    """
    Load DeBERTa-v3 with explicit LayerNorm key remapping.

    DeBERTa-v3 checkpoints use .beta/.gamma for LayerNorm parameters,
    but the HuggingFace DebertaV2 model class expects .weight/.bias.
    Without this fix, all LayerNorm layers load as random weights.
    """
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_config(config)

    pretrained = AutoModel.from_pretrained(model_name)
    pretrained_state = pretrained.state_dict()

    # Remap beta -> bias, gamma -> weight
    remapped_state = OrderedDict()
    for key, value in pretrained_state.items():
        new_key = key
        if '.LayerNorm.beta' in key:
            new_key = key.replace('.LayerNorm.beta', '.LayerNorm.bias')
        elif '.LayerNorm.gamma' in key:
            new_key = key.replace('.LayerNorm.gamma', '.LayerNorm.weight')
        remapped_state[new_key] = value

    model_state = model.state_dict()
    loaded_keys = []

    for key, value in remapped_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        elif f"deberta.{key}" in model_state and model_state[f"deberta.{key}"].shape == value.shape:
            model_state[f"deberta.{key}"] = value
            loaded_keys.append(f"deberta.{key}")

    model.load_state_dict(model_state)

    ln_loaded = sum(1 for k in loaded_keys if 'LayerNorm' in k)
    print(f"  Pretrained params mapped: {len(loaded_keys)}/{len(remapped_state)}, LayerNorm: {ln_loaded}")

    # Free the temporary model
    del pretrained, pretrained_state, remapped_state
    gc.collect()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

    return model


# ── Numerically Stable Focal Loss ────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss with label smoothing and class weight support.
    Numerically stable: clamps probabilities, guards against NaN.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.0,
                 class_weights: Optional[torch.Tensor] = None,
                 eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp().clamp(self.eps, 1.0 - self.eps)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.nll_loss(log_probs, targets, reduction='none')

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(self.eps, 1.0 - self.eps)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma

        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            focal_weight = focal_weight * cw[targets]

        loss = focal_weight * ce_loss

        if torch.isnan(loss).any():
            print("WARNING: NaN in focal loss, falling back to CE")
            loss = F.cross_entropy(logits, targets)

        return loss.mean()


# ── Custom Trainer ────────────────────────────────────────────────────────────

class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = self.focal_loss_fn(outputs.logits, labels)
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss


# ── Dataset ───────────────────────────────────────────────────────────────────

class AdviceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)


# ── Helper Functions ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_class_weights(labels: np.ndarray) -> List[float]:
    counts = np.bincount(labels, minlength=4).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (len(counts) * counts)
    weights = weights / weights.mean()
    return weights.tolist()


def balance_augmented_data(df: pd.DataFrame, label_col: str = 'type',
                            max_class0_ratio: float = 0.45) -> pd.DataFrame:
    """Downsample augmented class 0 to prevent majority-class dominance."""
    class0 = df[df[label_col] == 0]
    other = df[df[label_col] != 0]

    n_other = len(other)
    target_class0 = int(n_other * max_class0_ratio / (1.0 - max_class0_ratio))

    if len(class0) > target_class0 and 'source' in class0.columns:
        original_0 = class0[class0['source'] == 'original']
        augmented_0 = class0[class0['source'] != 'original']
        n_keep = max(target_class0 - len(original_0), 0)
        if n_keep < len(augmented_0):
            augmented_0 = augmented_0.sample(n=n_keep, random_state=42)
        class0 = pd.concat([original_0, augmented_0])
    elif len(class0) > target_class0:
        class0 = class0.sample(n=target_class0, random_state=42)

    return pd.concat([class0, other]).reset_index(drop=True)


def augment_and_balance(df: pd.DataFrame, config: TrainConfig) -> pd.DataFrame:
    """
    Augment training data and rebalance class 0.
    Called INSIDE each fold on the training split only — prevents data leakage.
    """
    augmented = run_augmentation_pipeline(df, passage_col='passage', label_col='type')
    balanced = balance_augmented_data(augmented, label_col='type',
                                       max_class0_ratio=config.max_class0_ratio)

    # Drop source column
    if 'source' in balanced.columns:
        balanced = balanced.drop(columns=['source'])

    return balanced


def load_raw_data(config: TrainConfig) -> pd.DataFrame:
    """Load and clean raw training data (no augmentation)."""
    df = pd.read_csv(config.data_path)
    df['passage'] = df['passage'].str.lower().str.strip()
    df['passage'] = df['passage'].str.replace(r"[^\w\s']", '', regex=True)
    df = df.dropna(subset=['type', 'passage'])
    df['type'] = df['type'].astype(int)
    return df


def encode_data(texts: List[str], tokenizer, max_length: int = 320):
    return tokenizer(
        texts, max_length=max_length, padding='max_length',
        truncation=True, return_tensors=None
    )


def free_memory():
    """Best-effort memory cleanup between folds."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


# ── Core Training Function ────────────────────────────────────────────────────

def train_model(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: TrainConfig,
    tokenizer,
    output_subdir: str = "final_model",
) -> Tuple[str, Optional[Dict], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Train a single model. Used for both k-fold validation and final training.

    Args:
        train_df: Training data (already augmented + balanced if applicable)
        val_df:   Validation data (None for final training on all data)
        config:   Training configuration
        tokenizer: Pretrained tokenizer
        output_subdir: Subdirectory name under config.output_dir

    Returns:
        (model_path, eval_metrics, val_logits_and_labels)
        val_logits_and_labels is (logits, labels) tuple or None if no val_df
    """
    output_dir = os.path.join(config.output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Encode
    train_encodings = encode_data(list(train_df['passage'].values), tokenizer, config.max_length)
    train_labels = train_df['type'].values.astype(int)
    train_dataset = AdviceDataset(train_encodings, train_labels)

    val_dataset = None
    if val_df is not None:
        val_encodings = encode_data(list(val_df['passage'].values), tokenizer, config.max_length)
        val_labels = val_df['type'].values.astype(int)
        val_dataset = AdviceDataset(val_encodings, val_labels)

    # Class weights
    class_weights = compute_class_weights(train_labels)
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights]}")

    # Load model
    model = load_deberta_v3_model(config.model_name, config.num_labels)
    device = get_device()
    print(f"  Device: {device}")

    # Metrics callback
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        pred_counts = np.bincount(preds, minlength=config.num_labels)
        print(f"    Predictions: {dict(zip(range(config.num_labels), pred_counts))}")
        return {
            'accuracy': (preds == labels).mean(),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
        }

    # Warmup steps
    steps_per_epoch = max(len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps), 1)
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    # Training arguments
    has_val = val_dataset is not None
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy='epoch' if has_val else 'no',
        save_strategy='epoch' if has_val else 'no',
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=warmup_steps,
        logging_steps=10,
        load_best_model_at_end=has_val,
        metric_for_best_model='eval_loss' if has_val else None,
        greater_is_better=False if has_val else None,
        save_total_limit=1,
        seed=config.seed,
        fp16=(device == "cuda"),
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_pin_memory=(device == "cuda"),
    )

    # Focal loss
    focal_loss_fn = FocalLoss(
        alpha=config.focal_alpha,
        gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing,
        class_weights=torch.tensor(class_weights, dtype=torch.float32),
    )

    # Callbacks
    callbacks = []
    if has_val:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        ))

    # Create trainer
    TrainerClass = FocalLossTrainer if config.use_focal_loss else Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics if has_val else None,
        callbacks=callbacks,
    )
    if config.use_focal_loss:
        trainer_kwargs['focal_loss_fn'] = focal_loss_fn
    trainer = TrainerClass(**trainer_kwargs)

    # Train
    trainer.train()

    # Evaluate
    eval_metrics = None
    val_logits_and_labels = None
    if has_val:
        eval_metrics = trainer.evaluate()
        print(f"  Results: {eval_metrics}")

        predictions = trainer.predict(val_dataset)
        val_logits = predictions.predictions  # raw logits, shape (n_val, n_classes)
        pred_labels = np.argmax(val_logits, axis=-1)
        val_logits_and_labels = (val_logits, val_labels)
        print(f"\n  Classification Report:")
        print(classification_report(
            val_labels, pred_labels,
            target_names=[config.label_names[i] for i in range(config.num_labels)],
            zero_division=0,
        ))

    # Save
    model_path = os.path.join(output_dir, "best_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    # Save val logits and metrics alongside model for deterministic resume
    if val_logits_and_labels is not None:
        np.savez(
            os.path.join(output_dir, "val_results.npz"),
            logits=val_logits_and_labels[0],
            labels=val_logits_and_labels[1],
        )
    if eval_metrics is not None:
        with open(os.path.join(output_dir, "eval_metrics.json"), 'w') as f:
            json.dump(eval_metrics, f, indent=2, default=str)

    # Free memory
    del trainer, model
    if val_dataset is not None:
        del val_dataset
    del train_dataset
    free_memory()

    return model_path, eval_metrics, val_logits_and_labels


# ── K-Fold Validation ────────────────────────────────────────────────────────

def validate_kfold(df: pd.DataFrame, config: TrainConfig, tokenizer) -> Tuple[List[Dict], Optional[str]]:
    """
    Run k-fold cross-validation to validate the architecture.

    Key: augmentation happens INSIDE each fold on the training split only.
    This prevents augmented variants of the same passage from appearing
    in both train and validation, which would cause data leakage and
    artificially inflate (and destabilize) fold metrics.

    Returns:
        (fold_results, calibration_path)
        calibration_path is the path to the fitted temperature scaler, or None.
    """
    print(f"\n{'='*60}")
    print(f"K-Fold Cross-Validation ({config.n_folds} folds)")
    print(f"{'='*60}")
    print(f"Raw data: {len(df)} samples\n")

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    fold_results = []
    all_oof_logits = []   # Out-of-fold logits for calibration
    all_oof_labels = []   # Corresponding true labels

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['type'])):
        print(f"\n{'─'*40}")
        print(f"Fold {fold + 1}/{config.n_folds}")
        print(f"{'─'*40}")

        raw_train = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        print(f"  Raw train:  {len(raw_train)} | Val: {len(val_df)}")
        print(f"  Val labels: {val_df['type'].value_counts().sort_index().to_dict()}")

        # Check if this fold was already completed (for resuming interrupted runs)
        fold_dir = os.path.join(config.output_dir, f"fold_{fold}")
        fold_model_path = os.path.join(fold_dir, "best_model")
        fold_results_path = os.path.join(fold_dir, "val_results.npz")
        fold_metrics_path = os.path.join(fold_dir, "eval_metrics.json")

        if os.path.exists(fold_model_path):
            print(f"  ✓ Fold already complete, loading saved results...")

            # Load saved logits and metrics (exact values from original training run)
            if os.path.exists(fold_results_path) and os.path.exists(fold_metrics_path):
                saved = np.load(fold_results_path)
                val_logits = saved['logits']
                val_labels = saved['labels']

                with open(fold_metrics_path, 'r') as f:
                    metrics = json.load(f)

                print(f"  Loaded saved metrics: acc={metrics.get('eval_accuracy', 0):.3f}, "
                      f"f1_macro={metrics.get('eval_f1_macro', 0):.3f}")
                fold_results.append(metrics)
                all_oof_logits.append(val_logits)
                all_oof_labels.append(val_labels)

            else:
                # Fallback: re-extract logits from saved model (older runs without saved results)
                print(f"  No saved results found, re-extracting logits from model...")
                saved_model = AutoModelForSequenceClassification.from_pretrained(
                    fold_model_path, local_files_only=True
                )
                saved_model.eval()
                val_encodings = encode_data(list(val_df['passage'].values), tokenizer, config.max_length)
                val_labels = val_df['type'].values.astype(int)
                val_dataset = AdviceDataset(val_encodings, val_labels)

                tmp_args = TrainingArguments(output_dir="/tmp/fold_resume", per_device_eval_batch_size=config.eval_batch_size)
                tmp_trainer = Trainer(model=saved_model, args=tmp_args)
                predictions = tmp_trainer.predict(val_dataset)
                val_logits = predictions.predictions
                pred_labels = np.argmax(val_logits, axis=-1)

                metrics = {
                    'eval_accuracy': float((pred_labels == val_labels).mean()),
                    'eval_f1_macro': float(f1_score(val_labels, pred_labels, average='macro', zero_division=0)),
                    'eval_f1_weighted': float(f1_score(val_labels, pred_labels, average='weighted', zero_division=0)),
                    'eval_precision_macro': float(precision_score(val_labels, pred_labels, average='macro', zero_division=0)),
                }
                print(f"  Re-extracted metrics: acc={metrics['eval_accuracy']:.3f}, "
                      f"f1_macro={metrics['eval_f1_macro']:.3f}")
                print(f"  WARNING: These may differ from original training metrics")
                fold_results.append(metrics)
                all_oof_logits.append(val_logits)
                all_oof_labels.append(val_labels)

                del saved_model, tmp_trainer
                free_memory()

            continue

        # Seed augmentation deterministically per fold — ensures identical results
        # whether this fold runs fresh or after earlier folds were skipped on resume.
        set_seed(config.seed + fold)

        # Augment ONLY the training split
        train_df = augment_and_balance(raw_train, config)
        print(f"  Augmented train: {len(train_df)}")
        print(f"  Train labels:    {train_df['type'].value_counts().sort_index().to_dict()}")

        # Use a different seed per fold for weight init diversity
        fold_config = TrainConfig(**{k: getattr(config, k) for k in config.__dataclass_fields__})
        fold_config.seed = config.seed + fold

        _, metrics, val_logits_and_labels = train_model(
            train_df, val_df, fold_config, tokenizer,
            output_subdir=f"fold_{fold}",
        )

        if metrics:
            fold_results.append(metrics)

        # Collect out-of-fold logits for calibration
        if val_logits_and_labels is not None:
            logits, labels = val_logits_and_labels
            all_oof_logits.append(logits)
            all_oof_labels.append(labels)

    # Summary
    print(f"\n{'='*60}")
    print(f"K-Fold Summary")
    print(f"{'='*60}")

    for metric in ['eval_accuracy', 'eval_f1_macro', 'eval_f1_weighted', 'eval_precision_macro']:
        values = [r.get(metric, 0) for r in fold_results]
        if values:
            print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Fit temperature scaling on pooled out-of-fold predictions
    calibration_path = None
    if all_oof_logits:
        print(f"\n{'='*60}")
        print(f"Calibration: Fitting Temperature Scaler")
        print(f"{'='*60}")

        pooled_logits = np.concatenate(all_oof_logits, axis=0)
        pooled_labels = np.concatenate(all_oof_labels, axis=0)
        print(f"  Pooled out-of-fold predictions: {len(pooled_labels)} samples")
        print(f"  Label distribution: {np.bincount(pooled_labels, minlength=config.num_labels).tolist()}")

        scaler = TemperatureScaler()
        temp = scaler.fit(pooled_logits, pooled_labels, verbose=True)

        calibration_path = os.path.join(config.output_dir, "temperature_scaler.pt")
        scaler.save(calibration_path)
        print(f"  Saved to: {calibration_path}")

    return fold_results, calibration_path


# ── Final Model Training ─────────────────────────────────────────────────────

def train_final_model(df: pd.DataFrame, config: TrainConfig, tokenizer) -> str:
    """
    Train the production model on 100% of the data.

    K-fold validated the architecture and hyperparameters.
    Now we use ALL available data to build the strongest possible model.
    No validation split — we train for the median number of epochs
    that worked well during k-fold (or just use early stopping disabled).
    """
    print(f"\n{'='*60}")
    print(f"Training Final Production Model")
    print(f"{'='*60}")
    print(f"Using all {len(df)} raw samples\n")

    # Seed for deterministic augmentation (independent of k-fold random state)
    set_seed(config.seed)

    # Augment the full dataset
    full_train = augment_and_balance(df, config)
    print(f"Augmented to {len(full_train)} samples")
    print(f"Label distribution: {full_train['type'].value_counts().sort_index().to_dict()}\n")

    # For the final model, we don't have a val set for early stopping.
    # Use a fixed epoch count — the median from k-fold, or config default.
    # We can also hold out a tiny slice (5%) just for loss monitoring.
    train_split, monitor_split = None, None
    if len(full_train) > 50:
        from sklearn.model_selection import train_test_split
        train_split, monitor_split = train_test_split(
            full_train, test_size=0.05, stratify=full_train['type'], random_state=config.seed
        )
        print(f"Training: {len(train_split)} | Monitor: {len(monitor_split)} (5% for loss tracking only)")
    else:
        train_split = full_train

    model_path, _, _ = train_model(
        train_split, monitor_split, config, tokenizer,
        output_subdir="final_model",
    )

    print(f"\nFinal model saved to: {model_path}")
    return model_path


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Set MPS memory behavior ──
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'

    config = TrainConfig(
        data_path="sample_training_data.csv",
        output_dir="models/deberta_distribution",
    )

    set_seed(config.seed)

    print(f"{'='*60}")
    print(f"Distribution Advice Model Training")
    print(f"{'='*60}\n")

    # Load raw data (no augmentation yet)
    df = load_raw_data(config)
    print(f"Loaded {len(df)} raw samples")
    print(f"Label distribution:\n{df['type'].value_counts().sort_index()}\n")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Step 1: K-fold validation (validates architecture, catches overfitting)
    fold_results, calibration_path = validate_kfold(df, config, tokenizer)

    # Save k-fold results
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "kfold_results.json"), 'w') as f:
        json.dump({
            'config': {k: str(v) for k, v in config.__dict__.items()},
            'fold_results': fold_results,
        }, f, indent=2, default=str)

    # Step 2: Train final model on all data
    final_path = train_final_model(df, config, tokenizer)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"K-fold results:    {config.output_dir}/kfold_results.json")
    print(f"Final model:       {final_path}")
    if calibration_path:
        print(f"Calibration file:  {calibration_path}")
    print(f"\nUse the final model + calibration file for inference.")
    print(f"K-fold models can be deleted.")
