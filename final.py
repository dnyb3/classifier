"""
SetFit + Ensemble Financial Advice Detection Pipeline
=====================================================
Architecture:
  1. Stage 1: Rule-based candidate extraction (existing module)
  2. Stage 2a: SetFit contrastive learning → fine-tuned sentence encoder + classification head
  3. Stage 2b: sklearn ensemble (SVM, LR, XGBoost) trained on SetFit embeddings
  4. Combined scoring: weighted blend of SetFit + ensemble predictions
  5. Streaming writes for watched calls during inference

The SetFit model serves dual purpose:
  - Its encoder produces domain-adapted embeddings (after contrastive fine-tuning)
  - Its classification head provides the primary prediction
  The sklearn ensemble operates on the SAME embeddings as a second opinion.

Dependencies:
  pip install setfit sentence-transformers xgboost scikit-learn torch pandas numpy tqdm datasets

Imports from existing modules:
  - stage1_extraction.extract_candidates  (candidate snippet extraction)
  - augment_training_data.run_augmentation_pipeline  (data augmentation)
"""

import os
import gc
import json
import time
import random
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer as SetFitTrainer, TrainingArguments as SetFitTrainingArguments
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed, ensemble will use RF fallback")
    from sklearn.ensemble import RandomForestClassifier

from stage1_extraction import extract_candidates
from augment_training_data import run_augmentation_pipeline

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Configuration for training and inference."""

    # ── Data ──
    data_path: str = "model_cases7.csv"
    passage_col: str = "passage"
    label_col: str = "type"
    num_labels: int = 4

    # ── SetFit ──
    setfit_base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    setfit_num_iterations: int = 20    # contrastive pairs per sample
    setfit_num_epochs: int = 1         # epochs over contrastive pairs
    setfit_body_learning_rate: float = 2e-5
    setfit_head_learning_rate: float = 1e-2
    setfit_batch_size: int = 16

    # ── Ensemble ──
    svm_C: float = 1.0
    lr_C: float = 1.0
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    calibrate_ensemble: bool = True

    # ── Combined scoring weights ──
    # SetFit head is the primary classifier; ensemble is the second opinion.
    setfit_weight: float = 0.55
    ensemble_weight: float = 0.45

    # ── Augmentation ──
    use_augmentation: bool = True
    max_class0_ratio: float = 0.45

    # ── K-fold ──
    n_folds: int = 5
    seed: int = 42

    # ── Paths ──
    output_dir: str = "models/setfit_ensemble"

    # ── Inference: Stage 1 ──
    words_before: int = 60
    words_after: int = 150
    min_preliminary_score: float = 0.15

    # ── Inference: thresholds ──
    advice_threshold: float = 0.35
    inference_batch_size: int = 256   # SetFit encoding is fast

    # ── Inference: streaming ──
    stream_output_csv: Optional[str] = None
    watch_column: Optional[str] = None
    watch_value: Optional[str] = None
    watch_threshold: float = 0.40
    top_k_output: int = 500

    # ── Labels ──
    label_names: Dict[int, str] = field(default_factory=lambda: {
        0: "no_advice",
        1: "roll_to_ira",
        2: "stay_in_plan",
        3: "roll_to_other_plan",
    })


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading & Augmentation
# ═════════════════════════════════════════════════════════════════════════════

def load_raw_data(config: PipelineConfig) -> pd.DataFrame:
    """Load and clean raw training data."""
    df = pd.read_csv(config.data_path)
    df[config.passage_col] = df[config.passage_col].str.lower().str.strip()
    df[config.passage_col] = df[config.passage_col].str.replace(r"[^\w\s']", '', regex=True)
    df = df.dropna(subset=[config.label_col, config.passage_col])
    df[config.label_col] = df[config.label_col].astype(int)
    logger.info(f"Loaded {len(df)} samples from {config.data_path}")
    logger.info(f"  Label distribution: {df[config.label_col].value_counts().sort_index().to_dict()}")
    return df


def augment_and_balance(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Augment training data and rebalance class 0."""
    if not config.use_augmentation:
        return df

    augmented = run_augmentation_pipeline(
        df, passage_col=config.passage_col, label_col=config.label_col
    )
    balanced = _balance_augmented(augmented, config.label_col, config.max_class0_ratio)

    if 'source' in balanced.columns:
        balanced = balanced.drop(columns=['source'])

    logger.info(f"  Augmented: {len(df)} → {len(balanced)} samples")
    return balanced


def _balance_augmented(df, label_col, max_class0_ratio):
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════════════════════
# SetFit Training
# ═════════════════════════════════════════════════════════════════════════════

def train_setfit_model(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: PipelineConfig,
) -> SetFitModel:
    """
    Train a SetFit model using contrastive learning.

    SetFit works in two phases:
      1. Fine-tunes the sentence transformer body with contrastive pairs
         (same-class pairs pushed together, different-class pulled apart)
      2. Trains a classification head on the resulting embeddings

    This is extremely data-efficient and fast — ideal for small labeled datasets.
    """
    logger.info(f"Training SetFit model (base: {config.setfit_base_model})")

    model = SetFitModel.from_pretrained(
        config.setfit_base_model,
        labels=list(config.label_names.values()),
    )

    # Prepare HuggingFace Datasets
    train_dataset = Dataset.from_dict({
        "text": train_df[config.passage_col].tolist(),
        "label": train_df[config.label_col].tolist(),
    })

    eval_dataset = None
    if val_df is not None and len(val_df) > 0:
        eval_dataset = Dataset.from_dict({
            "text": val_df[config.passage_col].tolist(),
            "label": val_df[config.label_col].tolist(),
        })

    training_args = SetFitTrainingArguments(
        output_dir=os.path.join(config.output_dir, "setfit_checkpoints"),
        num_iterations=config.setfit_num_iterations,
        num_epochs=config.setfit_num_epochs,
        body_learning_rate=config.setfit_body_learning_rate,
        head_learning_rate=config.setfit_head_learning_rate,
        batch_size=config.setfit_batch_size,
        seed=config.seed,
        evaluation_strategy="epoch" if eval_dataset else "no",
        logging_steps=50,
    )

    trainer = SetFitTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="f1",
    )

    trainer.train()

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        logger.info(f"  SetFit eval metrics: {metrics}")

    return model


def encode_with_setfit(model: SetFitModel, texts: List[str], batch_size: int = 256) -> np.ndarray:
    """
    Extract embeddings from the fine-tuned SetFit encoder.
    These embeddings live in a contrastive-optimized space where same-class
    samples cluster together — making them ideal features for sklearn classifiers.
    """
    # Access the sentence transformer body
    body = model.model_body
    embeddings = body.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


# ═════════════════════════════════════════════════════════════════════════════
# Sklearn Ensemble on SetFit Embeddings
# ═════════════════════════════════════════════════════════════════════════════

def train_sklearn_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: PipelineConfig = None,
) -> Dict[str, any]:
    """
    Train lightweight classifiers on SetFit embeddings.

    Why this works well:
      - SetFit contrastive learning arranges embeddings so classes are linearly
        separable → even simple classifiers achieve high accuracy.
      - SVM, LR, and XGBoost each have different inductive biases, so their
        agreement is a meaningful confidence signal.
      - Training takes seconds (embeddings are 384-dim, dataset is small).
    """
    config = config or PipelineConfig()
    models = {}

    # ── SVM with RBF kernel ──
    logger.info("  Training SVM...")
    svm_base = SVC(
        C=config.svm_C,
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=config.seed,
    )
    if config.calibrate_ensemble and X_val is not None:
        svm = CalibratedClassifierCV(svm_base, cv=3)
    else:
        svm = svm_base
    svm.fit(X_train, y_train)
    models['svm'] = svm

    # ── Logistic Regression ──
    logger.info("  Training Logistic Regression...")
    lr = LogisticRegression(
        C=config.lr_C,
        penalty='l2',
        class_weight='balanced',
        max_iter=1000,
        random_state=config.seed,
    )
    lr.fit(X_train, y_train)
    models['lr'] = lr

    # ── XGBoost or RF fallback ──
    if HAS_XGBOOST:
        logger.info("  Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=config.seed,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        models['xgb'] = xgb
    else:
        logger.info("  Training Random Forest (XGBoost not available)...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=config.seed,
        )
        rf.fit(X_train, y_train)
        models['rf'] = rf

    # ── Evaluate if val set available ──
    if X_val is not None and y_val is not None:
        _evaluate_ensemble(models, X_val, y_val)

    return models


def _evaluate_ensemble(models: Dict, X_val: np.ndarray, y_val: np.ndarray):
    """Evaluate each ensemble member and their combination."""
    for name, model in models.items():
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        logger.info(f"    {name}: acc={acc:.3f}, f1_macro={f1:.3f}")

    # Combined soft vote
    combined_probs = _ensemble_predict_proba(models, X_val)
    y_combined = np.argmax(combined_probs, axis=1)
    acc = accuracy_score(y_val, y_combined)
    f1 = f1_score(y_val, y_combined, average='macro', zero_division=0)
    logger.info(f"    ensemble_combined: acc={acc:.3f}, f1_macro={f1:.3f}")


def _ensemble_predict_proba(models: Dict, X: np.ndarray) -> np.ndarray:
    """Average probabilities across all ensemble members."""
    proba_list = []
    for model in models.values():
        proba_list.append(model.predict_proba(X))
    return np.mean(proba_list, axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# Combined Scorer
# ═════════════════════════════════════════════════════════════════════════════

class CombinedScorer:
    """
    Loads a trained SetFit model + sklearn ensemble and scores snippets.

    Scoring flow:
      1. Encode text with SetFit body → embedding (once)
      2. SetFit classification head → probabilities
      3. sklearn ensemble on same embedding → probabilities
      4. Weighted blend → final probabilities
      5. Agreement between SetFit and ensemble → confidence signal
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setfit_model = None
        self.ensemble_models = None
        self._load_models()

    def _load_models(self):
        """Load saved SetFit model and sklearn ensemble."""
        setfit_path = os.path.join(self.config.output_dir, "final_model", "setfit")
        ensemble_path = os.path.join(self.config.output_dir, "final_model", "ensemble.pkl")

        if os.path.exists(setfit_path):
            logger.info(f"Loading SetFit model from {setfit_path}")
            self.setfit_model = SetFitModel.from_pretrained(setfit_path)
        else:
            raise FileNotFoundError(f"SetFit model not found at {setfit_path}")

        if os.path.exists(ensemble_path):
            logger.info(f"Loading sklearn ensemble from {ensemble_path}")
            with open(ensemble_path, 'rb') as f:
                self.ensemble_models = pickle.load(f)
            logger.info(f"  Loaded {len(self.ensemble_models)} ensemble members: "
                        f"{list(self.ensemble_models.keys())}")
        else:
            logger.warning(f"Ensemble not found at {ensemble_path}, using SetFit only")
            self.ensemble_models = None

    def score_batch(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score a batch of texts.

        Returns:
            probs: shape (n, num_labels) — blended probabilities
            agreement: shape (n,) — fraction of classifiers agreeing on prediction
        """
        if not texts:
            return np.array([]), np.array([])

        # Step 1: Encode once with fine-tuned SetFit body
        embeddings = self.setfit_model.model_body.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

        # Step 2: SetFit head probabilities
        setfit_probs = self.setfit_model.predict_proba(texts)
        if isinstance(setfit_probs, torch.Tensor):
            setfit_probs = setfit_probs.cpu().numpy()
        setfit_probs = np.array(setfit_probs)

        # Ensure 2D
        if setfit_probs.ndim == 1:
            setfit_probs = setfit_probs.reshape(1, -1)

        # Step 3: Ensemble probabilities (if available)
        if self.ensemble_models:
            ens_probs = _ensemble_predict_proba(self.ensemble_models, embeddings)
            # Blend
            w_s = self.config.setfit_weight
            w_e = self.config.ensemble_weight
            blended = (w_s * setfit_probs + w_e * ens_probs) / (w_s + w_e)

            # Agreement: what fraction of all classifiers agree on the argmax?
            all_preds = [np.argmax(setfit_probs, axis=1)]
            for m in self.ensemble_models.values():
                all_preds.append(m.predict(embeddings))
            all_preds = np.stack(all_preds, axis=0)  # (n_classifiers, n_texts)
            consensus = np.argmax(blended, axis=1)
            agreement = (all_preds == consensus[np.newaxis, :]).mean(axis=0)
        else:
            blended = setfit_probs
            agreement = np.ones(len(texts))

        return blended, agreement

    def score_all(
        self,
        texts: List[str],
        batch_size: int = 256,
        on_batch_scored: Optional[callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Score all texts in batches with optional streaming callback."""
        if not texts:
            return np.array([]), np.array([])

        all_probs = [None] * len(texts)
        all_agreement = [None] * len(texts)

        # Sort by length for efficient batching
        indexed = list(enumerate(texts))
        indexed.sort(key=lambda x: len(x[1]))

        for start in tqdm(range(0, len(indexed), batch_size), desc="Scoring snippets"):
            batch = indexed[start:start + batch_size]
            orig_indices = [idx for idx, _ in batch]
            batch_texts = [text for _, text in batch]

            probs, agreement = self.score_batch(batch_texts)

            for j, oi in enumerate(orig_indices):
                all_probs[oi] = probs[j]
                all_agreement[oi] = agreement[j]

            if on_batch_scored is not None:
                on_batch_scored(orig_indices, probs, agreement)

        return np.stack(all_probs), np.array(all_agreement)


# ═════════════════════════════════════════════════════════════════════════════
# Training Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def train_pipeline(config: PipelineConfig):
    """
    Full training pipeline:
      1. Load and augment data
      2. K-fold validation (SetFit + ensemble)
      3. Train final model on all data
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Load data ──
    df = load_raw_data(config)
    texts = df[config.passage_col].values
    labels = df[config.label_col].values

    # ── K-fold validation ──
    logger.info(f"\n{'='*60}")
    logger.info(f"K-FOLD VALIDATION ({config.n_folds} folds)")
    logger.info("=" * 60)

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info(f"\n{'─'*40}")
        logger.info(f"Fold {fold + 1}/{config.n_folds}")
        logger.info(f"{'─'*40}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Check for completed fold
        fold_dir = os.path.join(config.output_dir, f"fold_{fold}")
        metrics_path = os.path.join(fold_dir, "eval_metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            logger.info(f"  ✓ Fold already complete — "
                        f"acc={metrics.get('accuracy', 0):.3f}, "
                        f"f1_macro={metrics.get('f1_macro', 0):.3f}")
            fold_results.append(metrics)
            continue

        os.makedirs(fold_dir, exist_ok=True)

        # Seed deterministically per fold
        set_seed(config.seed + fold)

        # Augment training split
        aug_train_df = augment_and_balance(train_df, config)
        logger.info(f"  Train: {len(train_df)} → {len(aug_train_df)} (augmented) | Val: {len(val_df)}")

        # Train SetFit
        setfit_model = train_setfit_model(aug_train_df, val_df, config)

        # Extract embeddings for ensemble
        logger.info("  Extracting embeddings for ensemble...")
        X_train = encode_with_setfit(setfit_model, aug_train_df[config.passage_col].tolist())
        X_val = encode_with_setfit(setfit_model, val_df[config.passage_col].tolist())
        y_train = aug_train_df[config.label_col].values
        y_val = val_df[config.label_col].values

        # Train ensemble
        logger.info("  Training sklearn ensemble on SetFit embeddings...")
        ensemble_models = train_sklearn_ensemble(X_train, y_train, X_val, y_val, config)

        # Combined evaluation
        setfit_probs = setfit_model.predict_proba(val_df[config.passage_col].tolist())
        if isinstance(setfit_probs, torch.Tensor):
            setfit_probs = setfit_probs.cpu().numpy()
        setfit_probs = np.array(setfit_probs)
        ens_probs = _ensemble_predict_proba(ensemble_models, X_val)

        w_s, w_e = config.setfit_weight, config.ensemble_weight
        combined_probs = (w_s * setfit_probs + w_e * ens_probs) / (w_s + w_e)
        combined_preds = np.argmax(combined_probs, axis=1)

        metrics = {
            'accuracy': float(accuracy_score(y_val, combined_preds)),
            'f1_macro': float(f1_score(y_val, combined_preds, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_val, combined_preds, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(y_val, combined_preds, average='macro', zero_division=0)),
        }

        logger.info(f"  Combined: acc={metrics['accuracy']:.3f}, f1_macro={metrics['f1_macro']:.3f}")
        logger.info(f"\n{classification_report(y_val, combined_preds, target_names=list(config.label_names.values()), zero_division=0)}")

        fold_results.append(metrics)

        # Save fold metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Clean up fold model to save memory (we only need the final model)
        del setfit_model, ensemble_models
        gc.collect()

    # K-fold summary
    logger.info(f"\n{'='*60}")
    logger.info("K-FOLD SUMMARY")
    logger.info("=" * 60)
    for metric in ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro']:
        vals = [r.get(metric, 0) for r in fold_results]
        logger.info(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ── Train final model on all data ──
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING FINAL MODEL (all data)")
    logger.info("=" * 60)

    final_dir = os.path.join(config.output_dir, "final_model")
    setfit_save_path = os.path.join(final_dir, "setfit")
    ensemble_save_path = os.path.join(final_dir, "ensemble.pkl")

    if os.path.exists(setfit_save_path) and os.path.exists(ensemble_save_path):
        logger.info("  ✓ Final model already trained, skipping")
    else:
        os.makedirs(final_dir, exist_ok=True)
        set_seed(config.seed)

        # Augment full dataset
        aug_df = augment_and_balance(df, config)
        logger.info(f"  Full dataset: {len(df)} → {len(aug_df)} (augmented)")

        # Train SetFit on everything
        setfit_model = train_setfit_model(aug_df, None, config)

        # Extract embeddings and train ensemble
        logger.info("  Extracting embeddings for final ensemble...")
        X_all = encode_with_setfit(setfit_model, aug_df[config.passage_col].tolist())
        y_all = aug_df[config.label_col].values
        ensemble_models = train_sklearn_ensemble(X_all, y_all, config=config)

        # Save
        setfit_model.save_pretrained(setfit_save_path)
        logger.info(f"  SetFit model saved to {setfit_save_path}")

        with open(ensemble_save_path, 'wb') as f:
            pickle.dump(ensemble_models, f)
        logger.info(f"  Ensemble saved to {ensemble_save_path}")

    # Save config
    config_path = os.path.join(config.output_dir, "pipeline_config.json")
    config_dict = {k: v for k, v in config.__dict__.items()
                   if not callable(v)}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# Inference Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_inference_pipeline(
    input_csv: str,
    output_csv: str,
    config: PipelineConfig,
    ranked_output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full inference pipeline:
      1. Load transcripts
      2. Stage 1: extract candidate snippets (rule-based)
      3. Stage 2: score with SetFit + ensemble
      4. Aggregate per-call results
      5. Stream watched calls during scoring
    """
    start_time = time.time()

    # ── Load data ──
    logger.info("Loading data...")
    df = pd.read_csv(input_csv, dtype={'INTERACTION_ID': str, 'CUS_ID': str, 'CALL_ID': str})
    logger.info(f"Loaded {len(df):,} transcripts")

    # ── Stage 1: Candidate Extraction ──
    logger.info(f"\n{'='*60}")
    logger.info("STAGE 1: Candidate Extraction")
    logger.info("=" * 60)

    stage1_start = time.time()
    all_candidates = []
    calls_with_candidates = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting candidates"):
        agent_text = str(row.get('AGENT_TRANSCRIPT', ''))
        if not agent_text or agent_text == 'nan':
            continue

        candidates = extract_candidates(
            agent_text,
            words_before=config.words_before,
            words_after=config.words_after,
            min_preliminary_score=config.min_preliminary_score,
        )

        if candidates:
            calls_with_candidates += 1
            for c in candidates:
                all_candidates.append({
                    'INTERACTION_ID': row['INTERACTION_ID'],
                    'CALL_ID': row.get('CALL_ID', ''),
                    'CUS_ID': row.get('CUS_ID', ''),
                    'INTERACTION_D': row.get('INTERACTION_D', ''),
                    'snippet_text': c.text,
                    'trigger_phrase': c.trigger_phrase,
                    'tier': c.tier,
                    'has_negation': c.has_negation,
                    'has_distribution_topic': c.has_distribution_topic,
                    'preliminary_score': c.preliminary_score,
                })

    stage1_time = time.time() - stage1_start

    if not all_candidates:
        logger.info("No candidates found.")
        return _empty_result(df, output_csv)

    candidates_df = pd.DataFrame(all_candidates)
    logger.info(f"\nStage 1 complete in {stage1_time:.1f}s")
    logger.info(f"  Calls with candidates: {calls_with_candidates:,} / {len(df):,}")
    logger.info(f"  Total snippets: {len(candidates_df):,}")
    logger.info(f"  Avg per flagged call: {len(candidates_df)/max(calls_with_candidates,1):.1f}")

    # ── Stage 2: Model Scoring ──
    logger.info(f"\n{'='*60}")
    logger.info("STAGE 2: SetFit + Ensemble Scoring")
    logger.info("=" * 60)

    stage2_start = time.time()
    scorer = CombinedScorer(config)

    # Set up streaming for watched calls
    stream_file = None
    stream_count = 0
    stream_header_written = False

    watch_ids = set()
    if config.watch_column and config.watch_value:
        if config.watch_column in df.columns:
            watch_ids = set(
                df.loc[df[config.watch_column].astype(str) == str(config.watch_value),
                       'INTERACTION_ID']
            )
            logger.info(f"  Watch filter: {config.watch_column}='{config.watch_value}' → "
                        f"{len(watch_ids):,} calls")

    if config.stream_output_csv:
        if not watch_ids:
            logger.warning("  stream_output_csv set but no watch filter configured. "
                           "Streaming disabled.")
        else:
            os.makedirs(os.path.dirname(config.stream_output_csv) or '.', exist_ok=True)
            stream_file = open(config.stream_output_csv, 'w', newline='')
            logger.info(f"  Streaming watched calls (threshold={config.watch_threshold}) "
                        f"to: {config.stream_output_csv}")

    def on_batch_scored(original_indices, batch_probs, batch_agreement):
        nonlocal stream_count, stream_header_written
        if stream_file is None:
            return

        for j, snippet_idx in enumerate(original_indices):
            row = candidates_df.iloc[snippet_idx]
            interaction_id = row['INTERACTION_ID']

            if interaction_id not in watch_ids:
                continue

            advice_probs_row = batch_probs[j, 1:]
            advice_score = float(advice_probs_row.max())
            advice_label = int(advice_probs_row.argmax()) + 1

            if advice_score >= config.watch_threshold:
                result_row = {
                    'INTERACTION_ID': interaction_id,
                    'CALL_ID': row.get('CALL_ID', ''),
                    'CUS_ID': row.get('CUS_ID', ''),
                    'advice_score': round(advice_score, 4),
                    'advice_label': config.label_names[advice_label],
                    'trigger_phrase': row['trigger_phrase'],
                    'tier': row['tier'],
                    'prob_no_advice': round(float(batch_probs[j, 0]), 4),
                    'prob_roll_to_ira': round(float(batch_probs[j, 1]), 4),
                    'prob_stay_in_plan': round(float(batch_probs[j, 2]), 4),
                    'prob_roll_to_other_plan': round(float(batch_probs[j, 3]), 4),
                    'snippet_preview': row['snippet_text'][:200],
                }
                pd.DataFrame([result_row]).to_csv(
                    stream_file, mode='a',
                    header=not stream_header_written, index=False,
                )
                stream_file.flush()
                stream_header_written = True
                stream_count += 1

                if stream_count <= 20 or stream_count % 50 == 0:
                    logger.info(f"  ⚡ Streamed #{stream_count}: {interaction_id} "
                                f"score={advice_score:.3f} → {config.label_names[advice_label]}")

    # Score all snippets
    snippet_texts = candidates_df['snippet_text'].tolist()
    probs, agreement = scorer.score_all(
        snippet_texts,
        batch_size=config.inference_batch_size,
        on_batch_scored=on_batch_scored if stream_file else None,
    )

    if stream_file:
        stream_file.close()
        logger.info(f"  Streaming complete: {stream_count:,} watched calls written")

    # Add scores to candidates
    candidates_df['ensemble_agreement'] = agreement

    for i, name in config.label_names.items():
        candidates_df[f'prob_{name}'] = probs[:, i]

    advice_probs = probs[:, 1:]
    candidates_df['advice_score'] = advice_probs.max(axis=1)
    candidates_df['advice_label'] = advice_probs.argmax(axis=1) + 1
    candidates_df['advice_label_name'] = candidates_df['advice_label'].map(config.label_names)

    candidates_df['combined_score'] = (
        candidates_df['advice_score'] *
        candidates_df['ensemble_agreement'] *
        (1 + candidates_df['preliminary_score']) / 2
    )

    stage2_time = time.time() - stage2_start
    logger.info(f"\nStage 2 complete in {stage2_time:.1f}s")

    # ── Aggregation ──
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATION: Per-Call Results")
    logger.info("=" * 60)

    best_per_call = (
        candidates_df
        .sort_values('combined_score', ascending=False)
        .groupby('INTERACTION_ID')
        .first()
        .reset_index()
    )

    best_per_call['predicted_label'] = np.where(
        best_per_call['advice_score'] >= config.advice_threshold,
        best_per_call['advice_label'],
        0
    )
    best_per_call['predicted_label_name'] = best_per_call['predicted_label'].map(config.label_names)

    candidate_counts = (
        candidates_df.groupby('INTERACTION_ID').size().reset_index(name='n_candidates')
    )

    # Merge back to original data
    merge_cols = [
        'INTERACTION_ID', 'predicted_label', 'predicted_label_name',
        'advice_score', 'combined_score', 'ensemble_agreement',
        'snippet_text', 'trigger_phrase', 'tier', 'has_negation',
        'has_distribution_topic',
        'prob_no_advice', 'prob_roll_to_ira',
        'prob_stay_in_plan', 'prob_roll_to_other_plan',
    ]
    result = df.merge(
        best_per_call[merge_cols], on='INTERACTION_ID', how='left'
    ).merge(
        candidate_counts, on='INTERACTION_ID', how='left'
    )

    # Fill calls with no candidates
    result['predicted_label'] = result['predicted_label'].fillna(0).astype(int)
    result['predicted_label_name'] = result['predicted_label_name'].fillna('no_advice')
    result['advice_score'] = result['advice_score'].fillna(0.0)
    result['combined_score'] = result['combined_score'].fillna(0.0)
    result['ensemble_agreement'] = result['ensemble_agreement'].fillna(0.0)
    result['n_candidates'] = result['n_candidates'].fillna(0).astype(int)
    result['snippet_text'] = result['snippet_text'].fillna('')
    result['trigger_phrase'] = result['trigger_phrase'].fillna('')
    result['tier'] = result['tier'].fillna(0).astype(int)
    result['has_negation'] = result['has_negation'].fillna(False)
    result['has_distribution_topic'] = result['has_distribution_topic'].fillna(False)

    result = result.rename(columns={
        'advice_score': 'call_score',
        'snippet_text': 'top_snippet',
        'tier': 'stage1_tier',
        'has_negation': 'negation_flag',
    })

    result = result.sort_values('combined_score', ascending=False).reset_index(drop=True)

    # ── Output ──
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    result.to_csv(output_csv, index=False)
    logger.info(f"\nFull results written to: {output_csv}")

    if ranked_output_csv:
        ranked = result[result['predicted_label'] != 0].head(config.top_k_output)
        os.makedirs(os.path.dirname(ranked_output_csv) or '.', exist_ok=True)
        ranked.to_csv(ranked_output_csv, index=False)
        logger.info(f"Review queue written to: {ranked_output_csv} ({len(ranked)} calls)")

    # ── Summary ──
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total transcripts:     {len(df):,}")
    logger.info(f"  Calls with candidates: {calls_with_candidates:,}")
    logger.info(f"  Calls with advice:     {(result['predicted_label'] != 0).sum():,}")
    for label, name in config.label_names.items():
        if label == 0:
            continue
        count = (result['predicted_label'] == label).sum()
        logger.info(f"    {name}: {count:,}")
    logger.info(f"\n  Stage 1 time: {stage1_time:.1f}s")
    logger.info(f"  Stage 2 time: {stage2_time:.1f}s")
    logger.info(f"  Total time:   {total_time:.1f}s ({total_time/60:.1f} minutes)")

    if config.stream_output_csv and stream_count > 0:
        logger.info(f"\n  Watched calls streamed: {stream_count:,} → {config.stream_output_csv}")

    return result


def _empty_result(df, output_csv):
    """Return empty result when no candidates found."""
    result = df.copy()
    result['predicted_label'] = 0
    result['predicted_label_name'] = 'no_advice'
    result['call_score'] = 0.0
    result['combined_score'] = 0.0
    result['ensemble_agreement'] = 0.0
    result['top_snippet'] = ''
    result['trigger_phrase'] = ''
    result['stage1_tier'] = 0
    result['negation_flag'] = False
    result['n_candidates'] = 0
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Entry Points
# ═════════════════════════════════════════════════════════════════════════════

def main_train():
    """Train the pipeline."""
    config = PipelineConfig(
        data_path="model_cases7.csv",
        output_dir="models/setfit_ensemble",
        use_augmentation=True,
        n_folds=5,
        seed=42,
    )
    train_pipeline(config)


def main_inference():
    """Run inference."""
    config = PipelineConfig(
        output_dir="models/setfit_ensemble",

        # Streaming: watch for unexpected advice in 'N' calls
        stream_output_csv="output/early_alerts.csv",
        watch_column="RECOMMENDATION_FLAG",
        watch_value="N",
        watch_threshold=0.40,

        top_k_output=500,
    )

    run_inference_pipeline(
        input_csv="data/monthly_transcripts.csv",
        output_csv="output/full_results.csv",
        ranked_output_csv="output/review_queue.csv",
        config=config,
    )


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'inference':
        main_inference()
    else:
        main_train()
