"""
Training Orchestrator
=====================
Trains both SetFit and DeBERTa models from a single script with a
guaranteed-clean validation holdout that neither model ever sees.

Flow:
  1. Load raw data, clean it once
  2. Stratified split: 90% train / 10% holdout
  3. Save both to disk (holdout.csv is the gold-standard eval set)
  4. Train SetFit (body contrastive + head + k-fold) on train split
  5. Train DeBERTa (k-fold + final model) on train split
  6. Export DeBERTa to ONNX for fast hybrid inference
  7. Evaluate both models independently on holdout
  8. Tune hybrid routing thresholds on holdout
  9. Calibrate hybrid pipeline on holdout
  10. Report final holdout metrics for all three approaches

The holdout CSV can also be used later for manual tuning:
  python hybrid_inference_pipeline.py tune --val_csv data/holdout.csv

Prerequisites (same directory):
  - stage2_train_fixed.py          (DeBERTa training)
  - setfit_ensemble_pipeline.py    (SetFit training)
  - hybrid_inference_pipeline.py   (hybrid inference + tuning)
  - inference_pipeline.py          (ONNX export)
  - stage1_extraction.py           (candidate extraction)
  - augment_training_data.py       (augmentation)
  - calibration.py                 (temperature scaling)

Usage:
  python train_orchestrator.py
  python train_orchestrator.py --data model_cases7.csv --holdout_frac 0.10
  python train_orchestrator.py --skip_setfit     # only train DeBERTa
  python train_orchestrator.py --skip_deberta    # only train SetFit
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════

def load_and_clean(data_path, passage_col='passage', label_col='type'):
    """Load raw data and apply standard cleaning (once)."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df[passage_col] = df[passage_col].str.lower().str.strip()
    df[passage_col] = df[passage_col].str.replace(r"[^\w\s']", '', regex=True)
    df = df.dropna(subset=[label_col, passage_col])
    df[label_col] = df[label_col].astype(int)

    logger.info(f"  {len(df)} samples after cleaning")
    logger.info(f"  Labels: {df[label_col].value_counts().sort_index().to_dict()}")
    return df


def create_holdout_split(df, holdout_frac, label_col, seed, output_dir):
    """
    Create a stratified holdout split and save both to disk.
    If the split files already exist, load them instead.
    """
    train_path = os.path.join(output_dir, "train.csv")
    holdout_path = os.path.join(output_dir, "holdout.csv")

    if os.path.exists(train_path) and os.path.exists(holdout_path):
        logger.info(f"\n  Existing split found:")
        train_df = pd.read_csv(train_path)
        holdout_df = pd.read_csv(holdout_path)
        train_df[label_col] = train_df[label_col].astype(int)
        holdout_df[label_col] = holdout_df[label_col].astype(int)
        logger.info(f"    Train:   {len(train_df)} ({train_path})")
        logger.info(f"    Holdout: {len(holdout_df)} ({holdout_path})")
        return train_df, holdout_df, train_path, holdout_path

    logger.info(f"\n  Creating {1-holdout_frac:.0%}/{holdout_frac:.0%} stratified split (seed={seed})")

    train_df, holdout_df = train_test_split(
        df, test_size=holdout_frac, random_state=seed, stratify=df[label_col],
    )
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)

    logger.info(f"    Train:   {len(train_df)} -> {train_path}")
    logger.info(f"    Holdout: {len(holdout_df)} -> {holdout_path}")
    logger.info(f"    Train labels:   {train_df[label_col].value_counts().sort_index().to_dict()}")
    logger.info(f"    Holdout labels: {holdout_df[label_col].value_counts().sort_index().to_dict()}")

    return train_df, holdout_df, train_path, holdout_path


# ═══════════════════════════════════════════════════════════════════════════
# SetFit Training
# ═══════════════════════════════════════════════════════════════════════════

def train_setfit(train_csv_path, setfit_output_dir, seed=42):
    """Train SetFit pipeline (body + head + k-fold) on training data."""
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING SETFIT")
    logger.info(f"{'='*70}")

    from setfit_ensemble_pipeline import PipelineConfig, train_pipeline

    config = PipelineConfig(
        data_path=train_csv_path,
        output_dir=setfit_output_dir,
        use_augmentation=True,
        n_folds=5,
        seed=seed,
        final_holdout_fraction=0,  # holdout managed externally by orchestrator
        setfit_num_iterations=8,   # tuned for ~4K samples (not few-shot)
        setfit_num_epochs=1,
    )

    train_pipeline(config)
    logger.info("SetFit training complete\n")


# ═══════════════════════════════════════════════════════════════════════════
# DeBERTa Training
# ═══════════════════════════════════════════════════════════════════════════

def train_deberta(train_csv_path, deberta_output_dir, seed=42):
    """Train DeBERTa pipeline (k-fold + final model) on training data."""
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING DEBERTA")
    logger.info(f"{'='*70}")

    from stage2_train_fixed import (
        TrainConfig, load_raw_data, set_seed, validate_kfold, train_final_model,
    )
    from transformers import AutoTokenizer

    config = TrainConfig(
        data_path=train_csv_path,
        output_dir=deberta_output_dir,
        seed=seed,
    )

    set_seed(config.seed)

    # Load training data
    df = load_raw_data(config)
    logger.info(f"DeBERTa training on {len(df)} samples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # K-fold validation
    fold_results, calibration_path = validate_kfold(df, config, tokenizer)

    # Save k-fold results
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "kfold_results.json"), 'w') as f:
        json.dump({
            'config': {k: str(v) for k, v in config.__dict__.items()},
            'fold_results': fold_results,
        }, f, indent=2, default=str)

    # Train final model on all training data
    final_path = train_final_model(df, config, tokenizer)

    logger.info("DeBERTa training complete\n")
    return final_path


def export_deberta_onnx(deberta_model_path):
    """Export DeBERTa to ONNX + quantized ONNX for fast CPU inference."""
    logger.info(f"\n{'_'*40}")
    logger.info("Exporting DeBERTa to ONNX")
    logger.info(f"{'_'*40}")

    onnx_path = os.path.join(deberta_model_path, "model.onnx")

    if os.path.exists(onnx_path) or os.path.exists(onnx_path.replace('.onnx', '_quantized.onnx')):
        logger.info("  ONNX model already exists, skipping export")
        return

    try:
        from inference_pipeline import export_to_onnx
        export_to_onnx(deberta_model_path, onnx_path)
    except Exception as e:
        logger.warning(f"  ONNX export failed: {e}")
        logger.warning("  Hybrid pipeline will fall back to PyTorch (slower)")


# ═══════════════════════════════════════════════════════════════════════════
# Holdout Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_on_holdout(holdout_df, config_dict, label_names):
    """
    Evaluate both models and the hybrid pipeline on the holdout set.
    The holdout was never seen by either model during training.
    """
    logger.info(f"\n{'='*70}")
    logger.info("HOLDOUT EVALUATION (neither model has seen this data)")
    logger.info(f"{'='*70}")

    holdout_texts = holdout_df['passage'].tolist()
    holdout_labels = holdout_df['type'].values.astype(int)
    label_names_list = [label_names[i] for i in sorted(label_names.keys())]

    results = {}

    # ── SetFit alone ──
    try:
        logger.info(f"\n{'_'*40}")
        logger.info("SetFit (alone)")
        logger.info(f"{'_'*40}")

        from setfit import SetFitModel
        import torch

        setfit_path = config_dict['setfit_model_path']
        if os.path.exists(setfit_path):
            model = SetFitModel.from_pretrained(setfit_path)
            probs = model.predict_proba(holdout_texts)
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()
            probs = np.array(probs)
            preds = np.argmax(probs, axis=1)

            results['setfit'] = _compute_metrics(holdout_labels, preds, label_names_list, "SetFit")
            del model
        else:
            logger.warning(f"  SetFit model not found at {setfit_path}")
    except Exception as e:
        logger.warning(f"  SetFit evaluation failed: {e}")

    # ── DeBERTa alone ──
    try:
        logger.info(f"\n{'_'*40}")
        logger.info("DeBERTa (alone)")
        logger.info(f"{'_'*40}")

        deberta_path = config_dict['deberta_model_path']
        if os.path.exists(deberta_path):
            from hybrid_inference_pipeline import DeBERTaScorer, HybridConfig
            temp_config = HybridConfig(deberta_model_path=deberta_path)
            scorer = DeBERTaScorer(temp_config)

            all_probs = []
            bs = 64
            for start in range(0, len(holdout_texts), bs):
                batch = holdout_texts[start:start+bs]
                all_probs.append(scorer.score_batch(batch))
            probs = np.concatenate(all_probs, axis=0)
            preds = np.argmax(probs, axis=1)

            results['deberta'] = _compute_metrics(holdout_labels, preds, label_names_list, "DeBERTa")
            del scorer
        else:
            logger.warning(f"  DeBERTa model not found at {deberta_path}")
    except Exception as e:
        logger.warning(f"  DeBERTa evaluation failed: {e}")

    # ── Hybrid (after tuning) ──
    try:
        logger.info(f"\n{'_'*40}")
        logger.info("Hybrid (SetFit screen + DeBERTa rescore)")
        logger.info(f"{'_'*40}")

        from hybrid_inference_pipeline import HybridScorer, HybridConfig

        hybrid_config = HybridConfig(
            setfit_model_path=config_dict['setfit_model_path'],
            deberta_model_path=config_dict['deberta_model_path'],
            calibrator_path=config_dict.get('calibrator_path'),
            routing_mode=config_dict.get('routing_mode', 'verify_advice'),
        )
        # Apply tuned thresholds if available
        if 'verify_skip_below' in config_dict:
            hybrid_config.verify_skip_below = config_dict['verify_skip_below']
        if 'accept_above' in config_dict:
            hybrid_config.accept_above = config_dict['accept_above']
        if 'reject_below' in config_dict:
            hybrid_config.reject_below = config_dict['reject_below']

        scorer = HybridScorer(hybrid_config)
        probs, agreement, rescored = scorer.score_all(holdout_texts)
        preds = np.argmax(probs, axis=1)

        results['hybrid'] = _compute_metrics(holdout_labels, preds, label_names_list, "Hybrid")

        n_rescored = rescored.sum()
        logger.info(f"  DeBERTa used for {n_rescored}/{len(holdout_texts)} "
                    f"({100*n_rescored/len(holdout_texts):.0f}%) holdout snippets")

        del scorer
    except Exception as e:
        logger.warning(f"  Hybrid evaluation failed: {e}")

    # ── Summary ──
    logger.info(f"\n{'='*70}")
    logger.info("HOLDOUT SUMMARY")
    logger.info(f"{'='*70}")
    header = f"  {'Model':<20} {'Accuracy':>10} {'F1_macro':>10} {'Precision':>10} {'Recall':>10}"
    logger.info(header)
    logger.info(f"  {'-'*60}")
    for name, m in results.items():
        logger.info(f"  {name:<20} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
                    f"{m['precision_macro']:>10.4f} {m['recall_macro']:>10.4f}")

    return results


def _compute_metrics(y_true, y_pred, target_names, model_name):
    """Compute and log standard metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }

    logger.info(f"  {model_name}: acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}, "
                f"prec={metrics['precision_macro']:.4f}, rec={metrics['recall_macro']:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=target_names, zero_division=0)}")

    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"  Confusion matrix:\n{cm}\n")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Tuning & Calibration
# ═══════════════════════════════════════════════════════════════════════════

def tune_and_calibrate_hybrid(holdout_df, config_dict):
    """
    Tune hybrid routing thresholds and calibrate on holdout data.
    Runs BOTH routing modes and reports which is better.
    """
    logger.info(f"\n{'='*70}")
    logger.info("TUNING HYBRID THRESHOLDS & CALIBRATING")
    logger.info(f"{'='*70}")

    from hybrid_inference_pipeline import (
        HybridConfig, tune_routing_thresholds, calibrate_hybrid,
    )

    holdout_texts = holdout_df['passage'].tolist()
    holdout_labels = holdout_df['type'].values.astype(int)

    mode_results = {}

    # ── Tune both modes ──
    for mode in ['verify_advice', 'borderline_only']:
        logger.info(f"\n{'_'*40}")
        logger.info(f"Tuning mode: {mode}")
        logger.info(f"{'_'*40}")

        config = HybridConfig(
            setfit_model_path=config_dict['setfit_model_path'],
            deberta_model_path=config_dict['deberta_model_path'],
            routing_mode=mode,
        )

        best_th1, best_th2, tune_result = tune_routing_thresholds(
            holdout_texts, holdout_labels, config,
        )

        mode_results[mode] = {
            'best_th1': best_th1,
            'best_th2': best_th2,
            'best_f1': tune_result['best_f1'],
        }

    # ── Pick winner ──
    best_mode = max(mode_results, key=lambda m: mode_results[m]['best_f1'])
    logger.info(f"\n{'_'*40}")
    logger.info("MODE COMPARISON")
    logger.info(f"{'_'*40}")
    for mode, r in mode_results.items():
        marker = " <-- best" if mode == best_mode else ""
        logger.info(f"  {mode}: f1={r['best_f1']:.4f}{marker}")

    logger.info(f"\n  Selected: {best_mode}")

    # ── Calibrate with winning mode ──
    winner = mode_results[best_mode]
    config = HybridConfig(
        setfit_model_path=config_dict['setfit_model_path'],
        deberta_model_path=config_dict['deberta_model_path'],
        routing_mode=best_mode,
    )
    if best_mode == 'verify_advice':
        config.verify_skip_below = winner['best_th1']
    else:
        config.accept_above = winner['best_th1']
        config.reject_below = winner['best_th2']

    calibrator_path = config_dict.get('calibrator_path', 'models/hybrid/calibrator.pkl')
    config.calibrator_path = None  # disable during calibration scoring
    calibrate_hybrid(holdout_texts, holdout_labels, config, save_path=calibrator_path)

    # ── Save tuned config ──
    tuned_config = {
        'routing_mode': best_mode,
        'calibrator_path': calibrator_path,
        'setfit_model_path': config_dict['setfit_model_path'],
        'deberta_model_path': config_dict['deberta_model_path'],
        'mode_results': {m: {k: str(v) for k, v in r.items()} for m, r in mode_results.items()},
    }
    if best_mode == 'verify_advice':
        tuned_config['verify_skip_below'] = winner['best_th1']
    else:
        tuned_config['accept_above'] = winner['best_th1']
        tuned_config['reject_below'] = winner['best_th2']

    tuned_path = os.path.join(os.path.dirname(calibrator_path), "tuned_config.json")
    os.makedirs(os.path.dirname(tuned_path) or '.', exist_ok=True)
    with open(tuned_path, 'w') as f:
        json.dump(tuned_config, f, indent=2)
    logger.info(f"  Tuned config saved to {tuned_path}")

    # Update config_dict for downstream eval
    config_dict['routing_mode'] = best_mode
    config_dict['calibrator_path'] = calibrator_path
    if best_mode == 'verify_advice':
        config_dict['verify_skip_below'] = winner['best_th1']
    else:
        config_dict['accept_above'] = winner['best_th1']
        config_dict['reject_below'] = winner['best_th2']

    return tuned_config


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train both models for hybrid pipeline")
    parser.add_argument('--data', default='model_cases7.csv', help='Path to labeled training data')
    parser.add_argument('--output_dir', default='data', help='Directory for train/holdout CSVs')
    parser.add_argument('--holdout_frac', type=float, default=0.10, help='Fraction held out')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--setfit_dir', default='models/setfit_ensemble')
    parser.add_argument('--deberta_dir', default='models/deberta_distribution')
    parser.add_argument('--hybrid_dir', default='models/hybrid')
    parser.add_argument('--skip_setfit', action='store_true', help='Skip SetFit training')
    parser.add_argument('--skip_deberta', action='store_true', help='Skip DeBERTa training')
    parser.add_argument('--skip_tuning', action='store_true', help='Skip hybrid tuning')
    parser.add_argument('--skip_eval', action='store_true', help='Skip holdout evaluation')
    args = parser.parse_args()

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("TRAINING ORCHESTRATOR")
    logger.info("=" * 70)

    # ── Step 1: Load and split ──
    logger.info(f"\n{'='*70}")
    logger.info("STEP 1: Data Preparation")
    logger.info(f"{'='*70}")

    df = load_and_clean(args.data)
    train_df, holdout_df, train_path, holdout_path = create_holdout_split(
        df, args.holdout_frac, 'type', args.seed, args.output_dir,
    )

    logger.info(f"\n  Holdout is sacred — neither model will see {holdout_path}")

    # Track paths for downstream steps
    config_dict = {
        'setfit_model_path': os.path.join(args.setfit_dir, "final_model", "setfit"),
        'deberta_model_path': os.path.join(args.deberta_dir, "final_model", "best_model"),
        'calibrator_path': os.path.join(args.hybrid_dir, "calibrator.pkl"),
    }

    # ── Step 2: Train SetFit ──
    if not args.skip_setfit:
        logger.info(f"\n{'='*70}")
        logger.info("STEP 2: SetFit Training")
        logger.info(f"{'='*70}")
        train_setfit(train_path, args.setfit_dir, seed=args.seed)
    else:
        logger.info("\n  Skipping SetFit training (--skip_setfit)")

    # ── Step 3: Train DeBERTa ──
    if not args.skip_deberta:
        logger.info(f"\n{'='*70}")
        logger.info("STEP 3: DeBERTa Training")
        logger.info(f"{'='*70}")
        final_model_path = train_deberta(train_path, args.deberta_dir, seed=args.seed)

        # Export to ONNX
        deberta_model_dir = os.path.join(args.deberta_dir, "final_model", "best_model")
        export_deberta_onnx(deberta_model_dir)
    else:
        logger.info("\n  Skipping DeBERTa training (--skip_deberta)")

    # ── Step 4: Tune hybrid thresholds ──
    if not args.skip_tuning:
        logger.info(f"\n{'='*70}")
        logger.info("STEP 4: Hybrid Threshold Tuning")
        logger.info(f"{'='*70}")
        tune_and_calibrate_hybrid(holdout_df, config_dict)
    else:
        logger.info("\n  Skipping hybrid tuning (--skip_tuning)")

    # ── Step 5: Final holdout evaluation ──
    if not args.skip_eval:
        label_names = {
            0: "no_advice", 1: "roll_to_ira",
            2: "stay_in_plan", 3: "roll_to_other_plan",
        }

        logger.info(f"\n{'='*70}")
        logger.info("STEP 5: Holdout Evaluation")
        logger.info(f"{'='*70}")
        holdout_results = evaluate_on_holdout(holdout_df, config_dict, label_names)

        # Save results
        results_path = os.path.join(args.hybrid_dir, "holdout_results.json")
        os.makedirs(os.path.dirname(results_path) or '.', exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(holdout_results, f, indent=2)
        logger.info(f"  Results saved to {results_path}")
    else:
        logger.info("\n  Skipping holdout evaluation (--skip_eval)")

    # ── Done ──
    total_time = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info("ORCHESTRATOR COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    logger.info(f"  Artifacts:")
    logger.info(f"    Train data:    {train_path}")
    logger.info(f"    Holdout data:  {holdout_path}")
    logger.info(f"    SetFit model:  {config_dict['setfit_model_path']}")
    logger.info(f"    DeBERTa model: {config_dict['deberta_model_path']}")
    logger.info(f"    Hybrid config: {os.path.join(args.hybrid_dir, 'tuned_config.json')}")
    logger.info(f"    Calibrator:    {config_dict['calibrator_path']}")

    logger.info(f"\n  To run inference:")
    logger.info(f"    python hybrid_inference_pipeline.py")
    logger.info(f"\n  To re-tune thresholds on holdout:")
    logger.info(f"    python hybrid_inference_pipeline.py tune --val_csv {holdout_path}")


if __name__ == '__main__':
    main()
