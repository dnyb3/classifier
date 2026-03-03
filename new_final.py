"""
SetFit + Ensemble Financial Advice Detection Pipeline
=====================================================
Architecture — four independent classification perspectives:

  Perspective 1 — SetFit head:
    Sentence transformer fine-tuned with contrastive learning → classification head.
    Sees: deep semantic meaning, paraphrase invariance.

  Perspective 2 — Embedding ensemble (SVM, LR, XGBoost):
    Trained on SetFit embeddings. Same semantic features, different decision boundaries.
    Sees: non-linear patterns in embedding space that SetFit's linear head misses.

  Perspective 3 — TF-IDF word n-gram model:
    TF-IDF (1,3)-grams → calibrated Logistic Regression. Completely independent of SetFit.
    Sees: surface-level word patterns, regulatory phrases, negation collocations.

  Perspective 4 — Character n-gram model:
    TF-IDF char (3,5)-grams → calibrated linear SVM. Completely independent of everything.
    Sees: sub-word patterns, robust to misspellings/ASR errors in messy transcripts.
    "recomend" still shares char trigrams with "recommend".

  Final blend → isotonic calibration → calibrated probabilities.

Dependencies:
  pip install setfit sentence-transformers xgboost scikit-learn torch pandas numpy tqdm datasets

Imports from existing modules:
  - stage1_extraction.extract_candidates
  - augment_training_data.run_augmentation_pipeline
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
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression

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


# Config
@dataclass
class PipelineConfig:
    data_path: str = "model_cases7.csv"
    passage_col: str = "passage"
    label_col: str = "type"
    num_labels: int = 4

    setfit_base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    setfit_num_iterations: int = 20
    setfit_num_epochs: int = 1
    setfit_body_learning_rate: float = 2e-5
    setfit_head_learning_rate: float = 1e-2
    setfit_batch_size: int = 16

    svm_C: float = 1.0
    lr_C: float = 1.0
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6

    tfidf_word_max_features: int = 20000
    tfidf_word_ngram_range: Tuple[int, int] = (1, 3)
    tfidf_char_max_features: int = 50000
    tfidf_char_ngram_range: Tuple[int, int] = (3, 5)

    weight_setfit: float = 0.35
    weight_embedding_ensemble: float = 0.25
    weight_tfidf_word: float = 0.20
    weight_tfidf_char: float = 0.20

    use_augmentation: bool = True
    max_class0_ratio: float = 0.45
    n_folds: int = 5
    seed: int = 42
    output_dir: str = "models/setfit_ensemble"

    words_before: int = 60
    words_after: int = 150
    min_preliminary_score: float = 0.15
    advice_threshold: float = 0.35
    inference_batch_size: int = 256

    stream_output_csv: Optional[str] = None
    watch_column: Optional[str] = None
    watch_value: Optional[str] = None
    watch_threshold: float = 0.40
    top_k_output: int = 500

    label_names: Dict[int, str] = field(default_factory=lambda: {
        0: "no_advice", 1: "roll_to_ira",
        2: "stay_in_plan", 3: "roll_to_other_plan",
    })


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading & Utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_data(config: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(config.data_path)
    df[config.passage_col] = df[config.passage_col].str.lower().str.strip()
    df[config.passage_col] = df[config.passage_col].str.replace(r"[^\w\s']", '', regex=True)
    df = df.dropna(subset=[config.label_col, config.passage_col])
    df[config.label_col] = df[config.label_col].astype(int)
    logger.info(f"Loaded {len(df)} samples from {config.data_path}")
    logger.info(f"  Labels: {df[config.label_col].value_counts().sort_index().to_dict()}")
    return df


def augment_and_balance(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if not config.use_augmentation:
        return df
    augmented = run_augmentation_pipeline(df, passage_col=config.passage_col, label_col=config.label_col)
    balanced = _balance_augmented(augmented, config.label_col, config.max_class0_ratio)
    if 'source' in balanced.columns:
        balanced = balanced.drop(columns=['source'])
    logger.info(f"  Augmented: {len(df)} -> {len(balanced)} samples")
    return balanced


def _balance_augmented(df, label_col, max_class0_ratio):
    class0 = df[df[label_col] == 0]
    other = df[df[label_col] != 0]
    target_class0 = int(len(other) * max_class0_ratio / (1.0 - max_class0_ratio))
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


def _get_setfit_probs(model: SetFitModel, texts: List[str]) -> np.ndarray:
    probs = model.predict_proba(texts)
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    probs = np.array(probs)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    return probs


def blend_all_probas(setfit_probs, ens_probs, tfidf_word_probs, char_probs, config):
    w = np.array([config.weight_setfit, config.weight_embedding_ensemble,
                  config.weight_tfidf_word, config.weight_tfidf_char])
    w = w / w.sum()
    return w[0]*setfit_probs + w[1]*ens_probs + w[2]*tfidf_word_probs + w[3]*char_probs


# ═══════════════════════════════════════════════════════════════════════════
# Perspective 1: SetFit Training
# ═══════════════════════════════════════════════════════════════════════════

def train_setfit_model(train_df, val_df, config):
    logger.info(f"  Training SetFit (base: {config.setfit_base_model})")
    model = SetFitModel.from_pretrained(
        config.setfit_base_model, labels=list(config.label_names.values()),
    )
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
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset, metric="f1",
    )
    trainer.train()
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        logger.info(f"    SetFit eval: {metrics}")
    return model


def encode_with_setfit(model, texts, batch_size=256):
    return model.model_body.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Perspective 2: Embedding Ensemble (SVM, LR, XGBoost)
# ═══════════════════════════════════════════════════════════════════════════

def train_embedding_ensemble(X_train, y_train, X_val=None, y_val=None, config=None):
    config = config or PipelineConfig()
    models = {}

    logger.info("    Training SVM on embeddings...")
    svm = CalibratedClassifierCV(
        SVC(C=config.svm_C, kernel='rbf', class_weight='balanced', random_state=config.seed), cv=3,
    )
    svm.fit(X_train, y_train)
    models['svm'] = svm

    logger.info("    Training LR on embeddings...")
    lr = LogisticRegression(
        C=config.lr_C, penalty='l2', class_weight='balanced', max_iter=1000, random_state=config.seed,
    )
    lr.fit(X_train, y_train)
    models['lr'] = lr

    if HAS_XGBOOST:
        logger.info("    Training XGBoost on embeddings...")
        xgb = XGBClassifier(
            n_estimators=config.xgb_n_estimators, max_depth=config.xgb_max_depth,
            learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss',
            random_state=config.seed, verbosity=0,
        )
        xgb.fit(X_train, y_train)
        models['xgb'] = xgb
    else:
        logger.info("    Training RF on embeddings (XGBoost unavailable)...")
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight='balanced', random_state=config.seed,
        )
        rf.fit(X_train, y_train)
        models['rf'] = rf

    if X_val is not None and y_val is not None:
        for name, m in models.items():
            preds = m.predict(X_val)
            logger.info(f"      {name}: acc={accuracy_score(y_val, preds):.3f}, "
                        f"f1={f1_score(y_val, preds, average='macro', zero_division=0):.3f}")
    return models


def _embedding_ensemble_proba(models, X):
    return np.mean([m.predict_proba(X) for m in models.values()], axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# Perspective 3: TF-IDF Word N-gram Model
# ═══════════════════════════════════════════════════════════════════════════

def train_tfidf_word_model(train_texts, y_train, val_texts=None, y_val=None, config=None):
    """
    TF-IDF word (1,3)-grams -> calibrated LR.
    Operates on raw surface patterns with zero semantic abstraction.
    Catches regulatory phrases like "not a recommendation", "cannot advise".
    Errors are uncorrelated with SetFit — when TF-IDF disagrees, it matters.
    """
    config = config or PipelineConfig()
    logger.info("    Training TF-IDF word n-gram model...")

    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=config.tfidf_word_ngram_range,
        max_features=config.tfidf_word_max_features,
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode',
    )
    X_train = vectorizer.fit_transform(train_texts)

    clf = CalibratedClassifierCV(
        LogisticRegression(C=1.0, penalty='l2', class_weight='balanced',
                           max_iter=2000, random_state=config.seed),
        cv=3,
    )
    clf.fit(X_train, y_train)

    if val_texts is not None and y_val is not None:
        X_val = vectorizer.transform(val_texts)
        preds = clf.predict(X_val)
        logger.info(f"      tfidf_word: acc={accuracy_score(y_val, preds):.3f}, "
                    f"f1={f1_score(y_val, preds, average='macro', zero_division=0):.3f}")

    return vectorizer, clf


# ═══════════════════════════════════════════════════════════════════════════
# Perspective 4: Character N-gram Model
# ═══════════════════════════════════════════════════════════════════════════

def train_char_ngram_model(train_texts, y_train, val_texts=None, y_val=None, config=None):
    """
    TF-IDF char (3,5)-grams -> calibrated linear SVM.

    ASR (speech-to-text) produces systematic misspellings:
      "recommend" -> "recomend", "rollover" -> "roll over"
    Word-level models treat each as a different token.
    Char trigrams share most features across variants:
      "recommend": rec,eco,com,omm,mme,men,end
      "recomend":  rec,eco,com,ome,men,end  (5/7 shared)

    Uses LinearSVC (fast on sparse 50K-dim features) with CalibratedClassifierCV.
    """
    config = config or PipelineConfig()
    logger.info("    Training character n-gram model...")

    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=config.tfidf_char_ngram_range,
        max_features=config.tfidf_char_max_features,
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_texts)

    clf = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight='balanced', max_iter=5000, random_state=config.seed),
        cv=3,
    )
    clf.fit(X_train, y_train)

    if val_texts is not None and y_val is not None:
        X_val = vectorizer.transform(val_texts)
        preds = clf.predict(X_val)
        logger.info(f"      char_ngram: acc={accuracy_score(y_val, preds):.3f}, "
                    f"f1={f1_score(y_val, preds, average='macro', zero_division=0):.3f}")

    return vectorizer, clf


# ═══════════════════════════════════════════════════════════════════════════
# Post-Blend Calibration
# ═══════════════════════════════════════════════════════════════════════════

class MulticlassCalibrator:
    """
    Isotonic regression calibrator for multiclass blended probabilities.

    Why calibrate after blending: the weighted average of 4 classifiers
    produces probabilities that don't map to empirical accuracy. A score
    of 0.70 might actually mean 85% or 55% real accuracy. Since the
    advice_threshold drives the keep/drop decision, miscalibration
    directly creates false positives or misses.

    Fits one isotonic regression per class on out-of-fold blended probs.
    Isotonic > Platt because the blend may have non-sigmoid calibration
    curves (it's a mixture of models, not a single logit output).
    """

    def __init__(self, num_labels=4):
        self.num_labels = num_labels
        self.calibrators = []

    def fit(self, blended_probs, true_labels):
        self.calibrators = []
        for c in range(self.num_labels):
            y_binary = (true_labels == c).astype(float)
            p_raw = blended_probs[:, c]
            iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            iso.fit(p_raw, y_binary)
            self.calibrators.append(iso)

        cal_probs = self.transform(blended_probs)
        raw_acc = accuracy_score(true_labels, np.argmax(blended_probs, axis=1))
        cal_acc = accuracy_score(true_labels, np.argmax(cal_probs, axis=1))

        logger.info(f"  Calibration fit on {len(true_labels)} OOF samples")
        logger.info(f"    Pre-calibration acc:  {raw_acc:.4f}")
        logger.info(f"    Post-calibration acc: {cal_acc:.4f}")

    def transform(self, blended_probs):
        if not self.calibrators:
            return blended_probs
        calibrated = np.zeros_like(blended_probs)
        for c, iso in enumerate(self.calibrators):
            calibrated[:, c] = iso.transform(blended_probs[:, c])
        row_sums = np.clip(calibrated.sum(axis=1, keepdims=True), 1e-8, None)
        return calibrated / row_sums

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.calibrators, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.calibrators = pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Combined Scorer (Inference)
# ═══════════════════════════════════════════════════════════════════════════

class CombinedScorer:
    """
    Loads all trained models and scores snippets with calibrated blended probs.

    Flow per batch:
      1. SetFit body encode -> embedding
      2. SetFit head -> probs_1
      3. SVM/LR/XGB on embedding -> probs_2
      4. TF-IDF word vectorizer -> LR -> probs_3
      5. Char n-gram vectorizer -> SVM -> probs_4
      6. Weighted blend -> raw probs
      7. Isotonic calibration -> calibrated probs
      8. Cross-perspective agreement -> confidence signal
    """

    def __init__(self, config):
        self.config = config
        self.setfit_model = None
        self.embedding_ensemble = None
        self.tfidf_word_vec = None
        self.tfidf_word_clf = None
        self.char_vec = None
        self.char_clf = None
        self.calibrator = None
        self._load_models()

    def _load_models(self):
        final_dir = os.path.join(self.config.output_dir, "final_model")

        setfit_path = os.path.join(final_dir, "setfit")
        if not os.path.exists(setfit_path):
            raise FileNotFoundError(f"SetFit not found at {setfit_path}")
        logger.info(f"Loading SetFit from {setfit_path}")
        self.setfit_model = SetFitModel.from_pretrained(setfit_path)

        ens_path = os.path.join(final_dir, "embedding_ensemble.pkl")
        if os.path.exists(ens_path):
            with open(ens_path, 'rb') as f:
                self.embedding_ensemble = pickle.load(f)
            logger.info(f"  Embedding ensemble: {list(self.embedding_ensemble.keys())}")

        tfidf_path = os.path.join(final_dir, "tfidf_word_model.pkl")
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                self.tfidf_word_vec, self.tfidf_word_clf = pickle.load(f)
            logger.info("  TF-IDF word model: loaded")

        char_path = os.path.join(final_dir, "char_ngram_model.pkl")
        if os.path.exists(char_path):
            with open(char_path, 'rb') as f:
                self.char_vec, self.char_clf = pickle.load(f)
            logger.info("  Char n-gram model: loaded")

        cal_path = os.path.join(final_dir, "calibrator.pkl")
        if os.path.exists(cal_path):
            self.calibrator = MulticlassCalibrator(self.config.num_labels)
            self.calibrator.load(cal_path)
            logger.info("  Calibrator: loaded")

        logger.info("  Scorer ready\n")

    def score_batch(self, texts):
        if not texts:
            return np.array([]), np.array([])

        # Perspective 1: SetFit head
        setfit_probs = _get_setfit_probs(self.setfit_model, texts)

        # Perspective 2: Embedding ensemble
        embeddings = None
        if self.embedding_ensemble:
            embeddings = self.setfit_model.model_body.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            ens_probs = _embedding_ensemble_proba(self.embedding_ensemble, embeddings)
        else:
            ens_probs = setfit_probs

        # Perspective 3: TF-IDF word
        if self.tfidf_word_vec and self.tfidf_word_clf:
            X_word = self.tfidf_word_vec.transform(texts)
            tfidf_word_probs = self.tfidf_word_clf.predict_proba(X_word)
        else:
            tfidf_word_probs = setfit_probs

        # Perspective 4: Char n-gram
        if self.char_vec and self.char_clf:
            X_char = self.char_vec.transform(texts)
            char_probs = self.char_clf.predict_proba(X_char)
        else:
            char_probs = setfit_probs

        # Blend
        blended = blend_all_probas(setfit_probs, ens_probs, tfidf_word_probs, char_probs, self.config)

        # Calibrate
        if self.calibrator:
            blended = self.calibrator.transform(blended)

        # Agreement across truly independent perspectives
        all_preds = [np.argmax(setfit_probs, axis=1)]
        if self.embedding_ensemble and embeddings is not None:
            for m in self.embedding_ensemble.values():
                all_preds.append(m.predict(embeddings))
        if self.tfidf_word_vec and self.tfidf_word_clf:
            all_preds.append(self.tfidf_word_clf.predict(X_word))
        if self.char_vec and self.char_clf:
            all_preds.append(self.char_clf.predict(X_char))

        all_preds = np.stack(all_preds, axis=0)
        consensus = np.argmax(blended, axis=1)
        agreement = (all_preds == consensus[np.newaxis, :]).mean(axis=0)

        return blended, agreement

    def score_all(self, texts, batch_size=256, on_batch_scored=None):
        if not texts:
            return np.array([]), np.array([])

        all_probs = [None] * len(texts)
        all_agreement = [None] * len(texts)

        indexed = sorted(enumerate(texts), key=lambda x: len(x[1]))

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


# ═══════════════════════════════════════════════════════════════════════════
# Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def _train_fold(train_df, val_df, config):
    """Train all 4 perspectives for one fold, return blended val probs."""
    train_texts = train_df[config.passage_col].tolist()
    val_texts = val_df[config.passage_col].tolist()
    y_train = train_df[config.label_col].values
    y_val = val_df[config.label_col].values

    # Perspective 1: SetFit
    setfit_model = train_setfit_model(train_df, val_df, config)

    # Perspective 2: Embedding ensemble
    logger.info("  Training embedding ensemble...")
    X_train_emb = encode_with_setfit(setfit_model, train_texts)
    X_val_emb = encode_with_setfit(setfit_model, val_texts)
    emb_ensemble = train_embedding_ensemble(X_train_emb, y_train, X_val_emb, y_val, config)

    # Perspective 3: TF-IDF word (independent of SetFit)
    logger.info("  Training text-based models (independent of SetFit)...")
    tfidf_word_vec, tfidf_word_clf = train_tfidf_word_model(
        train_texts, y_train, val_texts, y_val, config)

    # Perspective 4: Char n-gram (independent of everything)
    char_vec, char_clf = train_char_ngram_model(
        train_texts, y_train, val_texts, y_val, config)

    # Blend val predictions
    setfit_probs = _get_setfit_probs(setfit_model, val_texts)
    ens_probs = _embedding_ensemble_proba(emb_ensemble, X_val_emb)
    tfidf_word_probs = tfidf_word_clf.predict_proba(tfidf_word_vec.transform(val_texts))
    char_probs = char_clf.predict_proba(char_vec.transform(val_texts))
    blended = blend_all_probas(setfit_probs, ens_probs, tfidf_word_probs, char_probs, config)

    del setfit_model, emb_ensemble, tfidf_word_vec, tfidf_word_clf, char_vec, char_clf
    gc.collect()

    return blended, y_val


def train_pipeline(config):
    """
    Full training:
      1. K-fold (all 4 perspectives) -> collect OOF blended probs
      2. Fit calibrator on pooled OOF probs
      3. Train final models on all data
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    df = load_raw_data(config)
    texts = df[config.passage_col].values
    labels = df[config.label_col].values

    # K-fold validation
    logger.info(f"\n{'='*60}")
    logger.info(f"K-FOLD VALIDATION ({config.n_folds} folds)")
    logger.info("=" * 60)

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    fold_results = []
    all_oof_probs = []
    all_oof_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info(f"\n{'_'*40}")
        logger.info(f"Fold {fold + 1}/{config.n_folds}")
        logger.info(f"{'_'*40}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        fold_dir = os.path.join(config.output_dir, f"fold_{fold}")
        metrics_path = os.path.join(fold_dir, "eval_metrics.json")
        oof_path = os.path.join(fold_dir, "oof_results.npz")

        # Resume: load saved OOF results if available
        if os.path.exists(metrics_path) and os.path.exists(oof_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            saved = np.load(oof_path)
            all_oof_probs.append(saved['probs'])
            all_oof_labels.append(saved['labels'])
            logger.info(f"  Fold complete -- acc={metrics.get('accuracy', 0):.3f}, "
                        f"f1_macro={metrics.get('f1_macro', 0):.3f}")
            fold_results.append(metrics)
            continue

        os.makedirs(fold_dir, exist_ok=True)
        set_seed(config.seed + fold)

        aug_train_df = augment_and_balance(train_df, config)
        logger.info(f"  Train: {len(train_df)} -> {len(aug_train_df)} (aug) | Val: {len(val_df)}")

        blended_val_probs, y_val = _train_fold(aug_train_df, val_df, config)

        combined_preds = np.argmax(blended_val_probs, axis=1)
        metrics = {
            'accuracy': float(accuracy_score(y_val, combined_preds)),
            'f1_macro': float(f1_score(y_val, combined_preds, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_val, combined_preds, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(y_val, combined_preds, average='macro', zero_division=0)),
        }
        logger.info(f"\n  Combined (pre-cal): acc={metrics['accuracy']:.3f}, f1_macro={metrics['f1_macro']:.3f}")
        logger.info(f"\n{classification_report(y_val, combined_preds, target_names=list(config.label_names.values()), zero_division=0)}")

        fold_results.append(metrics)
        np.savez(oof_path, probs=blended_val_probs, labels=y_val)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        all_oof_probs.append(blended_val_probs)
        all_oof_labels.append(y_val)

    # K-fold summary
    logger.info(f"\n{'='*60}")
    logger.info("K-FOLD SUMMARY (pre-calibration)")
    logger.info("=" * 60)
    for metric in ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro']:
        vals = [r.get(metric, 0) for r in fold_results]
        logger.info(f"  {metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Fit calibrator
    logger.info(f"\n{'='*60}")
    logger.info("CALIBRATION")
    logger.info("=" * 60)

    final_dir = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    cal_path = os.path.join(final_dir, "calibrator.pkl")

    if all_oof_probs:
        pooled_probs = np.concatenate(all_oof_probs, axis=0)
        pooled_labels = np.concatenate(all_oof_labels, axis=0)
        calibrator = MulticlassCalibrator(config.num_labels)
        calibrator.fit(pooled_probs, pooled_labels)
        calibrator.save(cal_path)
        logger.info(f"  Saved to {cal_path}")

    # Train final models
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING FINAL MODEL (all data)")
    logger.info("=" * 60)

    setfit_save = os.path.join(final_dir, "setfit")
    emb_save = os.path.join(final_dir, "embedding_ensemble.pkl")
    tfidf_save = os.path.join(final_dir, "tfidf_word_model.pkl")
    char_save = os.path.join(final_dir, "char_ngram_model.pkl")

    if all(os.path.exists(p) for p in [setfit_save, emb_save, tfidf_save, char_save]):
        logger.info("  All final models already trained, skipping")
    else:
        set_seed(config.seed)
        aug_df = augment_and_balance(df, config)
        train_texts = aug_df[config.passage_col].tolist()
        y_all = aug_df[config.label_col].values

        setfit_model = train_setfit_model(aug_df, None, config)
        setfit_model.save_pretrained(setfit_save)
        logger.info(f"  SetFit saved")

        logger.info("  Training final embedding ensemble...")
        X_all_emb = encode_with_setfit(setfit_model, train_texts)
        emb_ens = train_embedding_ensemble(X_all_emb, y_all, config=config)
        with open(emb_save, 'wb') as f:
            pickle.dump(emb_ens, f)

        logger.info("  Training final TF-IDF word model...")
        tv, tc = train_tfidf_word_model(train_texts, y_all, config=config)
        with open(tfidf_save, 'wb') as f:
            pickle.dump((tv, tc), f)

        logger.info("  Training final char n-gram model...")
        cv, cc = train_char_ngram_model(train_texts, y_all, config=config)
        with open(char_save, 'wb') as f:
            pickle.dump((cv, cc), f)

    config_path = os.path.join(config.output_dir, "pipeline_config.json")
    config_dict = {k: v for k, v in config.__dict__.items() if not callable(v)}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# Inference Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_inference_pipeline(input_csv, output_csv, config, ranked_output_csv=None):
    start_time = time.time()

    logger.info("Loading data...")
    df = pd.read_csv(input_csv, dtype={'INTERACTION_ID': str, 'CUS_ID': str, 'CALL_ID': str})
    logger.info(f"Loaded {len(df):,} transcripts")

    # Stage 1: Candidate Extraction
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
    logger.info(f"\nStage 1: {stage1_time:.1f}s | {calls_with_candidates:,} calls, "
                f"{len(candidates_df):,} snippets")

    # Stage 2: Scoring
    logger.info(f"\n{'='*60}")
    logger.info("STAGE 2: Combined Scoring (4 perspectives + calibration)")
    logger.info("=" * 60)

    stage2_start = time.time()
    scorer = CombinedScorer(config)

    # Streaming setup
    stream_file = None
    stream_count = 0
    stream_header_written = False

    watch_ids = set()
    if config.watch_column and config.watch_value and config.watch_column in df.columns:
        watch_ids = set(
            df.loc[df[config.watch_column].astype(str) == str(config.watch_value), 'INTERACTION_ID']
        )
        logger.info(f"  Watch: {config.watch_column}='{config.watch_value}' -> {len(watch_ids):,} calls")

    if config.stream_output_csv and watch_ids:
        os.makedirs(os.path.dirname(config.stream_output_csv) or '.', exist_ok=True)
        stream_file = open(config.stream_output_csv, 'w', newline='')
        logger.info(f"  Streaming to: {config.stream_output_csv}")
    elif config.stream_output_csv:
        logger.warning("  stream_output_csv set but no watch filter. Streaming disabled.")

    def on_batch_scored(original_indices, batch_probs, batch_agreement):
        nonlocal stream_count, stream_header_written
        if stream_file is None:
            return
        for j, snippet_idx in enumerate(original_indices):
            row = candidates_df.iloc[snippet_idx]
            iid = row['INTERACTION_ID']
            if iid not in watch_ids:
                continue
            adv = batch_probs[j, 1:]
            advice_score = float(adv.max())
            advice_label = int(adv.argmax()) + 1
            if advice_score >= config.watch_threshold:
                result_row = {
                    'INTERACTION_ID': iid,
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
                    stream_file, mode='a', header=not stream_header_written, index=False,
                )
                stream_file.flush()
                stream_header_written = True
                stream_count += 1
                if stream_count <= 20 or stream_count % 50 == 0:
                    logger.info(f"  >> #{stream_count}: {iid} "
                                f"score={advice_score:.3f} -> {config.label_names[advice_label]}")

    probs, agreement = scorer.score_all(
        candidates_df['snippet_text'].tolist(),
        batch_size=config.inference_batch_size,
        on_batch_scored=on_batch_scored if stream_file else None,
    )

    if stream_file:
        stream_file.close()
        logger.info(f"  Streamed {stream_count:,} watched calls")

    # Score columns
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
    logger.info(f"\nStage 2: {stage2_time:.1f}s")

    # Aggregation
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATION")
    logger.info("=" * 60)

    best_per_call = (
        candidates_df.sort_values('combined_score', ascending=False)
        .groupby('INTERACTION_ID').first().reset_index()
    )
    best_per_call['predicted_label'] = np.where(
        best_per_call['advice_score'] >= config.advice_threshold,
        best_per_call['advice_label'], 0
    )
    best_per_call['predicted_label_name'] = best_per_call['predicted_label'].map(config.label_names)
    candidate_counts = candidates_df.groupby('INTERACTION_ID').size().reset_index(name='n_candidates')

    merge_cols = [
        'INTERACTION_ID', 'predicted_label', 'predicted_label_name',
        'advice_score', 'combined_score', 'ensemble_agreement',
        'snippet_text', 'trigger_phrase', 'tier', 'has_negation',
        'has_distribution_topic',
        'prob_no_advice', 'prob_roll_to_ira', 'prob_stay_in_plan', 'prob_roll_to_other_plan',
    ]
    result = df.merge(best_per_call[merge_cols], on='INTERACTION_ID', how='left')
    result = result.merge(candidate_counts, on='INTERACTION_ID', how='left')

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
        'advice_score': 'call_score', 'snippet_text': 'top_snippet',
        'tier': 'stage1_tier', 'has_negation': 'negation_flag',
    })
    result = result.sort_values('combined_score', ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    result.to_csv(output_csv, index=False)
    logger.info(f"\nResults: {output_csv}")

    if ranked_output_csv:
        ranked = result[result['predicted_label'] != 0].head(config.top_k_output)
        os.makedirs(os.path.dirname(ranked_output_csv) or '.', exist_ok=True)
        ranked.to_csv(ranked_output_csv, index=False)
        logger.info(f"Review queue: {ranked_output_csv} ({len(ranked)} calls)")

    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Transcripts: {len(df):,} | Candidates: {calls_with_candidates:,} | "
                f"Advice: {(result['predicted_label'] != 0).sum():,}")
    for label, name in config.label_names.items():
        if label == 0:
            continue
        logger.info(f"    {name}: {(result['predicted_label'] == label).sum():,}")
    logger.info(f"  Time: {total_time:.1f}s ({total_time/60:.1f}m)")

    return result


def _empty_result(df, output_csv):
    result = df.copy()
    for col, val in [('predicted_label', 0), ('predicted_label_name', 'no_advice'),
                     ('call_score', 0.0), ('combined_score', 0.0),
                     ('ensemble_agreement', 0.0), ('top_snippet', ''),
                     ('trigger_phrase', ''), ('stage1_tier', 0),
                     ('negation_flag', False), ('n_candidates', 0)]:
        result[col] = val
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Entry Points
# ═══════════════════════════════════════════════════════════════════════════

def main_train():
    config = PipelineConfig(
        data_path="model_cases7.csv",
        output_dir="models/setfit_ensemble",
        use_augmentation=True,
        n_folds=5,
        seed=42,
    )
    train_pipeline(config)


def main_inference():
    config = PipelineConfig(
        output_dir="models/setfit_ensemble",
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
