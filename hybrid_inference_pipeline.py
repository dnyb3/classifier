"""
Hybrid Inference Pipeline: SetFit (fast) + DeBERTa (accurate on hard cases)
===========================================================================

Architecture:
  1. Stage 1: Rule-based candidate extraction (fast, eliminates 70-85% of calls)
  2. Stage 2a: SetFit scores ALL snippets (~seconds for 280K snippets)
  3. Stage 2b: DeBERTa rescores ONLY borderline snippets (~15-30% of total)
  4. Merge: confident SetFit scores kept as-is, borderline replaced with DeBERTa
  5. Calibrate final blended probabilities

Why this works:
  SetFit (sentence transformer + contrastive learning) is fast because it
  mean-pools the entire snippet into one 384-dim vector. It's very good at
  the easy cases — clear advice, clear non-advice. But mean-pooling dilutes
  token-level signals: "I cannot recommend an IRA" and "I recommend an IRA"
  produce similar embeddings because the negation gets averaged away across
  200 tokens.

  DeBERTa (cross-encoder with token-level attention) is 20x slower but sees
  "cannot" directly modifying "recommend" through bidirectional attention.
  It excels exactly where SetFit struggles — the borderline cases where
  negation, hedging, or subtle phrasing determines the label.

  By routing only uncertain cases to DeBERTa, we get DeBERTa-level accuracy
  on hard cases and SetFit-level speed on easy cases. Total inference time
  is dominated by Stage 1 extraction + SetFit (fast) + DeBERTa on 15-30%
  of snippets (moderate).

Prerequisites:
  - Trained SetFit model: run setfit_ensemble_pipeline.py (just body + head)
  - Trained DeBERTa model: run stage2_train_fixed.py
  - Both models in their expected paths (see HybridConfig below)

Dependencies:
  pip install setfit sentence-transformers transformers torch onnxruntime
  pip install scikit-learn pandas numpy tqdm
"""

import os
import gc
import json
import time
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
from setfit import SetFitModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)

from stage1_extraction import extract_candidates

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HybridConfig:
    """Configuration for hybrid SetFit + DeBERTa inference."""

    # ── SetFit model ──
    setfit_model_path: str = "models/hybrid/setfit/final_model/setfit"

    # ── DeBERTa model ──
    deberta_model_path: str = "models/hybrid/deberta/final_model/best_model"
    deberta_max_length: int = 320
    deberta_use_onnx: bool = True
    deberta_calibration_path: Optional[str] = "models/hybrid/deberta/temperature_scaler.pt"

    # ── Routing ──
    # 'borderline_only': SetFit confident high or low → skip DeBERTa, middle → rescore
    # 'verify_advice':   Any SetFit advice signal → DeBERTa confirms, only confident
    #                    non-advice skips. Catches false positives at the cost of
    #                    more DeBERTa volume (~30-40% vs ~15-20%).
    routing_mode: str = 'verify_advice'  # 'borderline_only' or 'verify_advice'

    # ── Routing thresholds (borderline_only mode) ──
    # SetFit's max class probability determines routing:
    #   > accept_above:  SetFit is confident → accept its prediction (skip DeBERTa)
    #   < reject_below:  SetFit is very sure it's class 0 → accept (skip DeBERTa)
    #   between:         borderline → rescore with DeBERTa
    accept_above: float = 0.70   # high SetFit confidence → trust it
    reject_below: float = 0.12   # very low advice score → clearly not advice

    # ── Routing threshold (verify_advice mode) ──
    # Only snippets below this threshold skip DeBERTa. Everything above gets verified.
    # Lower = more snippets go to DeBERTa (safer). Higher = faster but misses more.
    verify_skip_below: float = 0.15

    # ── Blending for borderline cases ──
    # For snippets rescored by DeBERTa, how to combine:
    #   DeBERTa is primary (it's the stronger model on hard cases)
    #   but SetFit still provides a useful prior
    borderline_deberta_weight: float = 0.75
    borderline_setfit_weight: float = 0.25

    # ── Calibration ──
    calibrator_path: Optional[str] = "models/hybrid/calibrator.pkl"

    # ── Stage 1 settings ──
    words_before: int = 60
    words_after: int = 150
    min_preliminary_score: float = 0.15

    # ── Scoring ──
    advice_threshold: float = 0.35
    setfit_batch_size: int = 256   # SetFit is fast
    deberta_batch_size: int = 64   # DeBERTa needs smaller batches
    num_labels: int = 4
    top_k_output: int = 500

    # ── Streaming ──
    stream_output_csv: Optional[str] = None
    watch_column: Optional[str] = None
    watch_value: Optional[str] = None
    watch_threshold: float = 0.40

    # ── Labels ──
    label_names: Dict[int, str] = field(default_factory=lambda: {
        0: "no_advice", 1: "roll_to_ira",
        2: "stay_in_plan", 3: "roll_to_other_plan",
    })


# ═══════════════════════════════════════════════════════════════════════════
# Calibrator (same as setfit pipeline — shared utility)
# ═══════════════════════════════════════════════════════════════════════════

class MulticlassCalibrator:
    """Isotonic regression calibrator for multiclass probabilities."""

    def __init__(self, num_labels=4):
        self.num_labels = num_labels
        self.calibrators = []

    def fit(self, probs, true_labels):
        self.calibrators = []
        for c in range(self.num_labels):
            y_binary = (true_labels == c).astype(float)
            iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            iso.fit(probs[:, c], y_binary)
            self.calibrators.append(iso)

        raw_acc = accuracy_score(true_labels, np.argmax(probs, axis=1))
        cal_acc = accuracy_score(true_labels, np.argmax(self.transform(probs), axis=1))
        logger.info(f"  Calibrator fit: pre={raw_acc:.4f}, post={cal_acc:.4f}")

    def transform(self, probs):
        if not self.calibrators:
            return probs
        cal = np.zeros_like(probs)
        for c, iso in enumerate(self.calibrators):
            cal[:, c] = iso.transform(probs[:, c])
        sums = np.clip(cal.sum(axis=1, keepdims=True), 1e-8, None)
        return cal / sums

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.calibrators, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.calibrators = pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# DeBERTa Scorer (minimal — loads one model, scores batches)
# ═══════════════════════════════════════════════════════════════════════════

class DeBERTaScorer:
    """
    Loads a single DeBERTa model for rescoring borderline snippets.
    Supports ONNX Runtime for fast CPU inference.
    """

    def __init__(self, config: HybridConfig):
        self.config = config
        self.tokenizer = None
        self.onnx_session = None
        self.model = None
        self._load()

    def _load(self):
        model_path = self.config.deberta_model_path

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        # Try ONNX first
        if self.config.deberta_use_onnx:
            try:
                import onnxruntime as ort
                ort.set_default_logger_severity(3)

                model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
                onnx_quant = os.path.join(model_dir, "model_quantized.onnx")
                onnx_regular = os.path.join(model_dir, "model.onnx")

                onnx_file = None
                if os.path.exists(onnx_quant):
                    onnx_file = onnx_quant
                elif os.path.exists(onnx_regular):
                    onnx_file = onnx_regular

                if onnx_file:
                    sess_options = ort.SessionOptions()
                    sess_options.inter_op_num_threads = os.cpu_count()
                    sess_options.intra_op_num_threads = os.cpu_count()
                    self.onnx_session = ort.InferenceSession(onnx_file, sess_options)
                    logger.info(f"  DeBERTa loaded (ONNX): {onnx_file}")
                    return

            except ImportError:
                pass

        # PyTorch fallback
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        )
        self.model.eval()
        logger.info(f"  DeBERTa loaded (PyTorch): {model_path}")

    def score_batch(self, texts: List[str]) -> np.ndarray:
        """Score a batch, returns probs shape (n, num_labels)."""
        if not texts:
            return np.array([])

        encodings = self.tokenizer(
            texts,
            max_length=self.config.deberta_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt' if self.model else 'np',
        )

        if self.onnx_session:
            # Convert to numpy if needed
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.numpy()
                attention_mask = attention_mask.numpy()

            logits = self.onnx_session.run(
                ['logits'],
                {'input_ids': input_ids, 'attention_mask': attention_mask}
            )[0]
            probs = _softmax(logits)
        else:
            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=-1).numpy()

        return probs


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Scorer
# ═══════════════════════════════════════════════════════════════════════════

class HybridScorer:
    """
    Two-pass scorer: fast SetFit on everything, slow DeBERTa on selected cases.

    Two routing modes:

    'borderline_only' (fast, ~15-20% to DeBERTa):
      - advice_score >= accept_above:  confident advice → keep SetFit
      - advice_score <= reject_below:  confident no-advice → keep SetFit
      - between:                       borderline → rescore with DeBERTa

    'verify_advice' (safer, ~30-40% to DeBERTa):
      - advice_score <= verify_skip_below:  confident no-advice → keep SetFit
      - advice_score > verify_skip_below:   any advice signal → rescore with DeBERTa
      This catches false positives that 'borderline_only' would let through.

    For rescored cases, final probs are a weighted blend of DeBERTa (75%)
    and SetFit (25%).
    """

    def __init__(self, config: HybridConfig):
        self.config = config

        # Load SetFit
        logger.info(f"Loading SetFit from {config.setfit_model_path}")
        self.setfit = SetFitModel.from_pretrained(config.setfit_model_path)

        # Load DeBERTa
        logger.info(f"Loading DeBERTa from {config.deberta_model_path}")
        self.deberta = DeBERTaScorer(config)

        # Load calibrator
        self.calibrator = None
        if config.calibrator_path and os.path.exists(config.calibrator_path):
            self.calibrator = MulticlassCalibrator(config.num_labels)
            self.calibrator.load(config.calibrator_path)
            logger.info("  Hybrid calibrator: loaded")

        logger.info("  Hybrid scorer ready\n")

    def score_all(
        self,
        texts: List[str],
        on_batch_scored: Optional[callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Score all texts with hybrid routing.

        Returns:
            probs: (n, num_labels) — final calibrated probabilities
            agreement: (n,) — confidence signal (0.5 = models disagreed, 1.0 = agreed or confident)
            rescored: (n,) bool — True if DeBERTa was used for this snippet
        """
        n = len(texts)
        if n == 0:
            return np.array([]), np.array([]), np.array([], dtype=bool)

        # ── Pass 1: SetFit scores everything ──
        logger.info(f"  Pass 1: SetFit scoring {n:,} snippets...")
        t0 = time.time()
        setfit_probs = self._setfit_score_all(texts)
        setfit_time = time.time() - t0
        logger.info(f"    SetFit complete: {setfit_time:.1f}s "
                    f"({n / max(setfit_time, 0.01):.0f} snippets/sec)")

        # ── Route: identify which snippets go to DeBERTa ──
        advice_scores = setfit_probs[:, 1:].max(axis=1)

        if self.config.routing_mode == 'verify_advice':
            # Everything with any advice signal goes to DeBERTa.
            # Only confident non-advice skips. This catches false positives
            # that would otherwise sail through unchecked.
            skip_mask = advice_scores <= self.config.verify_skip_below
            rescore_mask = ~skip_mask
            rescore_indices = np.where(rescore_mask)[0]

            n_skip = skip_mask.sum()
            n_rescore = len(rescore_indices)
            pct = 100 * n_rescore / max(n, 1)

            logger.info(f"\n  Routing (verify_advice): "
                        f"{n_skip:,} confident no-advice ({100-pct:.0f}%) skip | "
                        f"{n_rescore:,} potential advice ({pct:.0f}%) -> DeBERTa")

        else:  # 'borderline_only'
            confident_high = advice_scores >= self.config.accept_above
            confident_low = advice_scores <= self.config.reject_below
            confident = confident_high | confident_low
            rescore_mask = ~confident
            rescore_indices = np.where(rescore_mask)[0]

            n_rescore = len(rescore_indices)
            pct = 100 * n_rescore / max(n, 1)

            logger.info(f"\n  Routing (borderline_only): "
                        f"{confident.sum():,} confident ({100-pct:.0f}%) | "
                        f"{n_rescore:,} borderline ({pct:.0f}%) -> DeBERTa")
            logger.info(f"    Confident high (>={self.config.accept_above}): "
                        f"{confident_high.sum():,}")
            logger.info(f"    Confident low (<={self.config.reject_below}): "
                        f"{confident_low.sum():,}")

        # ── Pass 2: DeBERTa rescores selected snippets ──
        final_probs = setfit_probs.copy()
        agreement = np.ones(n)  # non-rescored cases get 1.0
        rescored = np.zeros(n, dtype=bool)

        if n_rescore > 0:
            rescore_texts = [texts[i] for i in rescore_indices]

            logger.info(f"\n  Pass 2: DeBERTa rescoring {n_rescore:,} snippets...")
            t0 = time.time()
            deberta_probs = self._deberta_score_all(rescore_texts)
            deberta_time = time.time() - t0
            logger.info(f"    DeBERTa complete: {deberta_time:.1f}s "
                        f"({n_rescore / max(deberta_time, 0.01):.0f} snippets/sec)")

            # Blend: DeBERTa primary, SetFit as prior
            w_d = self.config.borderline_deberta_weight
            w_s = self.config.borderline_setfit_weight
            w_total = w_d + w_s

            for j, orig_idx in enumerate(rescore_indices):
                blended = (
                    w_d * deberta_probs[j] + w_s * setfit_probs[orig_idx]
                ) / w_total
                final_probs[orig_idx] = blended

                # Agreement: do SetFit and DeBERTa agree on the top class?
                setfit_pred = np.argmax(setfit_probs[orig_idx])
                deberta_pred = np.argmax(deberta_probs[j])
                agreement[orig_idx] = 1.0 if setfit_pred == deberta_pred else 0.5

            # Log agreement stats
            rescore_agreement = agreement[rescore_indices]
            agree_pct = 100 * (rescore_agreement == 1.0).mean()
            logger.info(f"    Agreement: {agree_pct:.0f}% "
                        f"({(rescore_agreement == 1.0).sum():,}/{n_rescore:,})")

            rescored[rescore_indices] = True

        # ── Calibrate ──
        if self.calibrator:
            final_probs = self.calibrator.transform(final_probs)

        # ── Fire streaming callback in two phases ──
        if on_batch_scored is not None:
            # Phase 1: non-rescored results (SetFit only, available immediately)
            setfit_only_indices = np.where(~rescored)[0].tolist()
            if setfit_only_indices:
                on_batch_scored(
                    setfit_only_indices,
                    final_probs[setfit_only_indices],
                    agreement[setfit_only_indices],
                )

            # Phase 2: rescored results (available after DeBERTa)
            if n_rescore > 0:
                rescore_list = rescore_indices.tolist()
                on_batch_scored(
                    rescore_list,
                    final_probs[rescore_indices],
                    agreement[rescore_indices],
                )

        return final_probs, agreement, rescored

    def _setfit_score_all(self, texts: List[str]) -> np.ndarray:
        """Score all texts with SetFit in batches."""
        all_probs = []
        bs = self.config.setfit_batch_size

        for start in range(0, len(texts), bs):
            batch = texts[start:start + bs]
            probs = self.setfit.predict_proba(batch)
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()
            probs = np.array(probs)
            if probs.ndim == 1:
                probs = probs.reshape(1, -1)
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def _deberta_score_all(self, texts: List[str]) -> np.ndarray:
        """Score texts with DeBERTa in batches."""
        all_probs = []
        bs = self.config.deberta_batch_size

        for start in tqdm(range(0, len(texts), bs), desc="DeBERTa rescoring"):
            batch = texts[start:start + bs]
            probs = self.deberta.score_batch(batch)
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# Threshold Tuning
# ═══════════════════════════════════════════════════════════════════════════

def tune_routing_thresholds(
    val_texts: List[str],
    val_labels: np.ndarray,
    config: HybridConfig,
    accept_candidates: List[float] = None,
    reject_candidates: List[float] = None,
    verify_candidates: List[float] = None,
) -> Tuple[float, float, Dict]:
    """
    Find optimal routing thresholds on a validation set.

    For 'borderline_only': tests combinations of accept_above and reject_below.
    For 'verify_advice': tests verify_skip_below thresholds.

    Returns:
        (best_accept_or_verify, best_reject_or_0, results_dict)
    """
    if accept_candidates is None:
        accept_candidates = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    if reject_candidates is None:
        reject_candidates = [0.08, 0.10, 0.12, 0.15, 0.20]
    if verify_candidates is None:
        verify_candidates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

    logger.info(f"Tuning routing thresholds (mode: {config.routing_mode})...")

    # Load scorer just to get access to both models
    # Temporarily disable calibration so we see raw routing behavior
    orig_cal = config.calibrator_path
    config.calibrator_path = None
    scorer = HybridScorer(config)
    config.calibrator_path = orig_cal

    # Score everything with both models (full pass for simulation)
    logger.info("  Scoring val set with SetFit...")
    setfit_probs = scorer._setfit_score_all(val_texts)

    logger.info("  Scoring val set with DeBERTa (full — for threshold simulation)...")
    deberta_probs = scorer._deberta_score_all(val_texts)

    advice_scores = setfit_probs[:, 1:].max(axis=1)

    w_d = config.borderline_deberta_weight
    w_s = config.borderline_setfit_weight
    w_total = w_d + w_s

    best_f1 = -1
    best_th1 = config.accept_above
    best_th2 = config.reject_below
    results = []

    if config.routing_mode == 'verify_advice':
        # Single threshold: everything above it goes to DeBERTa
        for skip_th in verify_candidates:
            final = setfit_probs.copy()
            rescore = advice_scores > skip_th
            n_rescore = rescore.sum()

            if n_rescore > 0:
                final[rescore] = (
                    w_d * deberta_probs[rescore] + w_s * setfit_probs[rescore]
                ) / w_total

            preds = np.argmax(final, axis=1)
            f1 = f1_score(val_labels, preds, average='macro', zero_division=0)
            acc = accuracy_score(val_labels, preds)
            prec = precision_score(val_labels, preds, average='macro', zero_division=0)
            pct_deberta = 100 * n_rescore / len(val_labels)

            results.append({
                'verify_skip_below': skip_th,
                'f1_macro': f1, 'accuracy': acc, 'precision_macro': prec,
                'pct_deberta': pct_deberta, 'n_rescore': int(n_rescore),
            })

            if f1 > best_f1:
                best_f1 = f1
                best_th1 = skip_th
                best_th2 = 0.0  # not used in this mode

        results_df = pd.DataFrame(results).sort_values('f1_macro', ascending=False)
        logger.info(f"\n  Top 5 verify_advice thresholds:")
        for _, row in results_df.head(5).iterrows():
            logger.info(f"    skip_below={row['verify_skip_below']:.2f}: "
                        f"f1={row['f1_macro']:.4f} acc={row['accuracy']:.4f} "
                        f"prec={row['precision_macro']:.4f} deberta={row['pct_deberta']:.0f}%")

        logger.info(f"\n  Best: verify_skip_below={best_th1}, f1_macro={best_f1:.4f}")

    else:  # borderline_only
        for accept_th in accept_candidates:
            for reject_th in reject_candidates:
                if reject_th >= accept_th:
                    continue

                final = setfit_probs.copy()
                rescore = (advice_scores < accept_th) & (advice_scores > reject_th)
                n_rescore = rescore.sum()

                if n_rescore > 0:
                    final[rescore] = (
                        w_d * deberta_probs[rescore] + w_s * setfit_probs[rescore]
                    ) / w_total

                preds = np.argmax(final, axis=1)
                f1 = f1_score(val_labels, preds, average='macro', zero_division=0)
                acc = accuracy_score(val_labels, preds)
                prec = precision_score(val_labels, preds, average='macro', zero_division=0)
                pct_deberta = 100 * n_rescore / len(val_labels)

                results.append({
                    'accept_above': accept_th, 'reject_below': reject_th,
                    'f1_macro': f1, 'accuracy': acc, 'precision_macro': prec,
                    'pct_deberta': pct_deberta, 'n_rescore': int(n_rescore),
                })

                if f1 > best_f1:
                    best_f1 = f1
                    best_th1 = accept_th
                    best_th2 = reject_th

        results_df = pd.DataFrame(results).sort_values('f1_macro', ascending=False)
        logger.info(f"\n  Top 5 borderline_only threshold combos:")
        for _, row in results_df.head(5).iterrows():
            logger.info(f"    accept>{row['accept_above']:.2f} reject<{row['reject_below']:.2f}: "
                        f"f1={row['f1_macro']:.4f} acc={row['accuracy']:.4f} "
                        f"prec={row['precision_macro']:.4f} deberta={row['pct_deberta']:.0f}%")

        logger.info(f"\n  Best: accept_above={best_th1}, reject_below={best_th2}, "
                    f"f1_macro={best_f1:.4f}")

    return best_th1, best_th2, {'results': results, 'best_f1': best_f1}


def calibrate_hybrid(
    val_texts: List[str],
    val_labels: np.ndarray,
    config: HybridConfig,
    save_path: str = None,
) -> MulticlassCalibrator:
    """
    Fit a calibrator on hybrid predictions from a validation set.
    Runs the full hybrid scoring (with routing) and fits isotonic regression
    on the resulting blended probabilities.
    """
    logger.info("Calibrating hybrid pipeline...")

    # Temporarily disable calibration during scoring
    orig_cal = config.calibrator_path
    config.calibrator_path = None

    scorer = HybridScorer(config)
    probs, _, _ = scorer.score_all(val_texts)

    config.calibrator_path = orig_cal

    calibrator = MulticlassCalibrator(config.num_labels)
    calibrator.fit(probs, val_labels)

    save_path = save_path or config.calibrator_path or "models/hybrid/calibrator.pkl"
    calibrator.save(save_path)
    logger.info(f"  Calibrator saved to {save_path}")

    return calibrator


# ═══════════════════════════════════════════════════════════════════════════
# Inference Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_inference_pipeline(input_csv, output_csv, config, ranked_output_csv=None):
    """
    Full hybrid inference pipeline:
      1. Load transcripts
      2. Stage 1: extract candidate snippets
      3. Stage 2: hybrid SetFit + DeBERTa scoring with routing
      4. Aggregate per-call results
      5. Stream watched calls
    """
    start_time = time.time()

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
    logger.info(f"\nStage 1: {stage1_time:.1f}s | {calls_with_candidates:,} calls, "
                f"{len(candidates_df):,} snippets")

    # ── Stage 2: Hybrid Scoring ──
    logger.info(f"\n{'='*60}")
    logger.info("STAGE 2: Hybrid Scoring (SetFit screen -> DeBERTa rescore)")
    logger.info("=" * 60)

    stage2_start = time.time()
    scorer = HybridScorer(config)

    # Set up streaming
    stream_file = None
    stream_count = 0
    stream_header_written = False

    watch_ids = set()
    if config.watch_column and config.watch_value and config.watch_column in df.columns:
        watch_ids = set(
            df.loc[df[config.watch_column].astype(str) == str(config.watch_value), 'INTERACTION_ID']
        )
        if watch_ids:
            logger.info(f"  Watch: {config.watch_column}='{config.watch_value}' -> "
                        f"{len(watch_ids):,} calls")

    if config.stream_output_csv and watch_ids:
        os.makedirs(os.path.dirname(config.stream_output_csv) or '.', exist_ok=True)
        stream_file = open(config.stream_output_csv, 'w', newline='')
        logger.info(f"  Streaming to: {config.stream_output_csv}")

    def on_batch_scored(original_indices, batch_probs, batch_agreement):
        nonlocal stream_count, stream_header_written
        if stream_file is None:
            return
        for j, snippet_idx in enumerate(original_indices):
            row_data = candidates_df.iloc[snippet_idx]
            iid = row_data['INTERACTION_ID']
            if iid not in watch_ids:
                continue
            adv = batch_probs[j, 1:]
            advice_score = float(adv.max())
            advice_label = int(adv.argmax()) + 1
            if advice_score >= config.watch_threshold:
                result_row = {
                    'INTERACTION_ID': iid,
                    'CALL_ID': row_data.get('CALL_ID', ''),
                    'CUS_ID': row_data.get('CUS_ID', ''),
                    'advice_score': round(advice_score, 4),
                    'advice_label': config.label_names[advice_label],
                    'trigger_phrase': row_data['trigger_phrase'],
                    'tier': row_data['tier'],
                    # Note: scored_by in streaming uses agreement as proxy —
                    # may show 'setfit' for rescored snippets where both models
                    # agreed. The final CSV's scored_by column is always correct.
                    'scored_by': 'deberta' if batch_agreement[j] < 1.0 else 'setfit',
                    'prob_no_advice': round(float(batch_probs[j, 0]), 4),
                    'prob_roll_to_ira': round(float(batch_probs[j, 1]), 4),
                    'prob_stay_in_plan': round(float(batch_probs[j, 2]), 4),
                    'prob_roll_to_other_plan': round(float(batch_probs[j, 3]), 4),
                    'snippet_preview': row_data['snippet_text'][:200],
                }
                pd.DataFrame([result_row]).to_csv(
                    stream_file, mode='a', header=not stream_header_written, index=False,
                )
                stream_file.flush()
                stream_header_written = True
                stream_count += 1
                if stream_count <= 20 or stream_count % 50 == 0:
                    logger.info(f"  >> #{stream_count}: {iid} score={advice_score:.3f}")

    # Score
    snippet_texts = candidates_df['snippet_text'].tolist()
    probs, agreement, rescored = scorer.score_all(
        snippet_texts,
        on_batch_scored=on_batch_scored if stream_file else None,
    )

    if stream_file:
        stream_file.close()
        logger.info(f"  Streamed {stream_count:,} watched calls")

    stage2_time = time.time() - stage2_start

    # ── Add scores to candidates ──
    candidates_df['ensemble_agreement'] = agreement
    candidates_df['scored_by'] = np.where(rescored, 'deberta', 'setfit')

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

    logger.info(f"\nStage 2: {stage2_time:.1f}s")
    scored_by = candidates_df['scored_by'].value_counts()
    logger.info(f"  Scored by: {scored_by.to_dict()}")

    # ── Aggregation ──
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
        'advice_score', 'combined_score', 'ensemble_agreement', 'scored_by',
        'snippet_text', 'trigger_phrase', 'tier', 'has_negation',
        'has_distribution_topic',
        'prob_no_advice', 'prob_roll_to_ira', 'prob_stay_in_plan', 'prob_roll_to_other_plan',
    ]
    result = df.merge(best_per_call[merge_cols], on='INTERACTION_ID', how='left')
    result = result.merge(candidate_counts, on='INTERACTION_ID', how='left')

    # Fill calls with no candidates
    result['predicted_label'] = result['predicted_label'].fillna(0).astype(int)
    result['predicted_label_name'] = result['predicted_label_name'].fillna('no_advice')
    result['advice_score'] = result['advice_score'].fillna(0.0)
    result['combined_score'] = result['combined_score'].fillna(0.0)
    result['ensemble_agreement'] = result['ensemble_agreement'].fillna(0.0)
    result['scored_by'] = result['scored_by'].fillna('none')
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

    # ── Output ──
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    result.to_csv(output_csv, index=False)
    logger.info(f"\nResults: {output_csv}")

    if ranked_output_csv:
        ranked = result[result['predicted_label'] != 0].head(config.top_k_output)
        os.makedirs(os.path.dirname(ranked_output_csv) or '.', exist_ok=True)
        ranked.to_csv(ranked_output_csv, index=False)
        logger.info(f"Review queue: {ranked_output_csv} ({len(ranked)} calls)")

    # ── Summary ──
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Transcripts:  {len(df):,}")
    logger.info(f"  Candidates:   {calls_with_candidates:,} calls, {len(candidates_df):,} snippets")
    logger.info(f"  Advice found: {(result['predicted_label'] != 0).sum():,}")
    for label, name in config.label_names.items():
        if label == 0:
            continue
        logger.info(f"    {name}: {(result['predicted_label'] == label).sum():,}")

    # Scoring path breakdown
    advice_calls = result[result['predicted_label'] != 0]
    if len(advice_calls) > 0:
        by_scorer = advice_calls['scored_by'].value_counts()
        logger.info(f"\n  Advice calls by scorer: {by_scorer.to_dict()}")

    logger.info(f"\n  Timing:")
    logger.info(f"    Stage 1 (extraction): {stage1_time:.1f}s")
    logger.info(f"    Stage 2 (scoring):    {stage2_time:.1f}s")
    logger.info(f"    Total:                {total_time:.1f}s ({total_time/60:.1f}m)")

    return result


def _empty_result(df, output_csv):
    result = df.copy()
    for col, val in [('predicted_label', 0), ('predicted_label_name', 'no_advice'),
                     ('call_score', 0.0), ('combined_score', 0.0),
                     ('ensemble_agreement', 0.0), ('scored_by', 'none'),
                     ('top_snippet', ''), ('trigger_phrase', ''),
                     ('stage1_tier', 0), ('negation_flag', False),
                     ('n_candidates', 0)]:
        result[col] = val
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Entry Points
# ═══════════════════════════════════════════════════════════════════════════

def main_inference():
    # Load tuned config if available, otherwise use defaults
    tuned_path = "models/hybrid/tuned_config.json"
    if os.path.exists(tuned_path):
        logger.info(f"Loading tuned config from {tuned_path}")
        with open(tuned_path, 'r') as f:
            tuned = json.load(f)
        config = HybridConfig(
            setfit_model_path=tuned.get('setfit_model_path', 'models/setfit_ensemble/final_model/setfit'),
            deberta_model_path=tuned.get('deberta_model_path', 'models/deberta_distribution/final_model/best_model'),
            routing_mode=tuned.get('routing_mode', 'verify_advice'),
            calibrator_path=tuned.get('calibrator_path', 'models/hybrid/calibrator.pkl'),
            deberta_use_onnx=True,
            stream_output_csv="output/early_alerts.csv",
            watch_column="RECOMMENDATION_FLAG",
            watch_value="N",
            watch_threshold=0.40,
        )
        if 'verify_skip_below' in tuned:
            config.verify_skip_below = float(tuned['verify_skip_below'])
        if 'accept_above' in tuned:
            config.accept_above = float(tuned['accept_above'])
        if 'reject_below' in tuned:
            config.reject_below = float(tuned['reject_below'])
    else:
        logger.info("No tuned config found, using defaults (verify_advice mode)")
        config = HybridConfig(
            setfit_model_path="models/setfit_ensemble/final_model/setfit",
            deberta_model_path="models/deberta_distribution/final_model/best_model",
            routing_mode='verify_advice',
            deberta_use_onnx=True,
            stream_output_csv="output/early_alerts.csv",
            watch_column="RECOMMENDATION_FLAG",
            watch_value="N",
            watch_threshold=0.40,
        )

    run_inference_pipeline(
        input_csv="data/monthly_transcripts.csv",
        output_csv="output/full_results.csv",
        ranked_output_csv="output/review_queue.csv",
        config=config,
    )


def main_tune():
    """
    Tune routing thresholds and calibrate on holdout data.
    Usage:
      python hybrid_inference_pipeline.py tune --val_csv data/holdout.csv
      python hybrid_inference_pipeline.py tune --val_csv data/holdout.csv --mode verify_advice
      python hybrid_inference_pipeline.py tune --val_csv data/holdout.csv --mode both
    """
    import sys
    import argparse

    argv = sys.argv[2:]  # skip script name and 'tune'

    parser = argparse.ArgumentParser(description="Tune hybrid routing thresholds")
    parser.add_argument('--val_csv', required=True, help='Holdout CSV with passage and type columns')
    parser.add_argument('--passage_col', default='passage')
    parser.add_argument('--label_col', default='type')
    parser.add_argument('--setfit_path', default='models/setfit_ensemble/final_model/setfit')
    parser.add_argument('--deberta_path', default='models/deberta_distribution/final_model/best_model')
    parser.add_argument('--mode', default='both', choices=['verify_advice', 'borderline_only', 'both'],
                        help='Which routing mode(s) to tune')
    args = parser.parse_args(argv)

    val_df = pd.read_csv(args.val_csv)
    val_texts = val_df[args.passage_col].tolist()
    val_labels = val_df[args.label_col].values.astype(int)

    logger.info(f"Loaded holdout: {len(val_df)} samples from {args.val_csv}")
    logger.info(f"  Labels: {pd.Series(val_labels).value_counts().sort_index().to_dict()}")

    modes = ['verify_advice', 'borderline_only'] if args.mode == 'both' else [args.mode]
    mode_results = {}

    for mode in modes:
        logger.info(f"\n{'_'*40}")
        logger.info(f"Tuning: {mode}")
        logger.info(f"{'_'*40}")

        config = HybridConfig(
            setfit_model_path=args.setfit_path,
            deberta_model_path=args.deberta_path,
            routing_mode=mode,
        )

        best_th1, best_th2, results = tune_routing_thresholds(
            val_texts, val_labels, config
        )
        mode_results[mode] = {'th1': best_th1, 'th2': best_th2, 'f1': results['best_f1']}

    # Pick best mode
    best_mode = max(mode_results, key=lambda m: mode_results[m]['f1'])
    winner = mode_results[best_mode]

    logger.info(f"\n{'='*40}")
    logger.info("RESULTS")
    logger.info(f"{'='*40}")
    for mode, r in mode_results.items():
        marker = " <-- best" if mode == best_mode else ""
        logger.info(f"  {mode}: f1={r['f1']:.4f}{marker}")

    # Calibrate with winning mode
    config = HybridConfig(
        setfit_model_path=args.setfit_path,
        deberta_model_path=args.deberta_path,
        routing_mode=best_mode,
    )
    if best_mode == 'verify_advice':
        config.verify_skip_below = winner['th1']
    else:
        config.accept_above = winner['th1']
        config.reject_below = winner['th2']

    calibrate_hybrid(val_texts, val_labels, config)

    # Save
    tuned = {
        'routing_mode': best_mode,
        'setfit_model_path': args.setfit_path,
        'deberta_model_path': args.deberta_path,
        'calibrator_path': config.calibrator_path,
        'best_f1': winner['f1'],
        'holdout_csv': args.val_csv,
        'holdout_size': len(val_df),
    }
    if best_mode == 'verify_advice':
        tuned['verify_skip_below'] = winner['th1']
    else:
        tuned['accept_above'] = winner['th1']
        tuned['reject_below'] = winner['th2']

    tuned_path = "models/hybrid/tuned_config.json"
    os.makedirs(os.path.dirname(tuned_path) or '.', exist_ok=True)
    with open(tuned_path, 'w') as f:
        json.dump(tuned, f, indent=2)

    logger.info(f"\nTuned config saved to {tuned_path}")
    logger.info(f"Winner: {best_mode} (f1={winner['f1']:.4f})")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'tune':
        main_tune()
    else:
        main_inference()
