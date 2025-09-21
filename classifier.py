"""
advice_detector_pipeline.py

Two-stage advice detection pipeline:
- Stage 1: binary advice vs no-advice (high precision tuned)
- Stage 2: multiclass classification on positives (IRA, stay, plan-to-plan)

How to use:
1) Install requirements (see bottom).
2) Put your training CSV at `training_data/model_cases7.csv` with columns:
   - 'passage' (text snippet, ~180-200 words currently)
   - 'type' (0=no advice, 1=IRA, 2=stay, 3=plan-to-plan)
3) Run:
   python advice_detector_pipeline.py --train
   This trains embedding model + classifiers and writes models to ./models/
4) Run inference:
   python advice_detector_pipeline.py --infer --transcript-dir path/to/transcripts --out results.csv

Notes:
- CPU-only assumed. This uses sentence-transformers (all-MiniLM-L6-v2) which is compact and effective.
- You can replace the encoder name in CONFIG['encoder_model_name'] to another SBERT if needed.
- Tune CONFIG thresholds for desired top-N precision.
"""

import os
import re
import argparse
import math
import json
from pathlib import Path
from typing import List, Tuple, Dict
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load

# ML libs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV

# Optional: LightGBM (fall back to sklearn if not installed)
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# Sentence Transformer
from sentence_transformers import SentenceTransformer, util

################################################################################
# CONFIG
################################################################################
CONFIG = {
    # encoder: compact SBERT recommended for CPU (all-MiniLM-L6-v2 is ~22M params)
    'encoder_model_name': 'all-MiniLM-L6-v2',

    # length: how many tokens/words around match to extract; keep compatible with your current ~160 words
    'left_words': 40,
    'right_words': 120,
    'max_snippet_words': 400,  # safety cap

    # regex improvements: include more patterns and also negation capture
    'advice_keywords': [
        r'\brecommend(s|ed|ing)?\b',
        r'\bsuggestion(s)?\b',
        r'\bsuggest(s|ed|ing)?\b',
        r'\bpropose(s|d|d)?\b',
        r'\badvice\b',
        r'\burg(e|ed|ing)?\b',
        r'\b(you should|you might|you may want)\b',
        r'\bbest (option|bet)\b',
        r'\badvise(s|d|ing)?\b',
        r'\bshould consider\b',
        r'\bmy (opinion|input)\b',
        r'\bencourage(s|d)?\b',
        r'\bdirection\b',
        r'\bconclusion\b',
        r'\bjudgement\b',
        # add common idioms
        r'\bmove(ing)? (your|the)? (money|funds|balance)\b',
        r'\broll(ing|over)? (to|into)?\b',
        r'\bro(?:ll)?over\b',
        r'\bIRA\b',
        r'\brohlen?'  # placeholder for common typos if needed
    ],

    # negation words to help generate negative examples / filter false positives
    'negation_words': [r'\bnot\b', r"\bdon'?t\b", r'\bno\b', r'\bcannot\b', r'\bcan\'t\b', r'\bunable\b', r'\bno recommendation\b'],

    # model and persistence
    'model_dir': './models',
    'binary_clf_name': 'binary_advice_clf.joblib',
    'multiclass_clf_name': 'multiclass_advice_clf.joblib',
    'encoder_name': 'sbert_encoder.joblib',
    'label_map': {0: 'no_advice', 1: 'roll_to_ira', 2: 'stay_in_plan', 3: 'roll_to_plan'},

    # training/eval
    'test_size': 0.10,
    'random_state': 26,
    'binary_clf_type': 'logreg',  # 'logreg' or 'lgb'
    'binary_c': 1.0,  # inverse regularization for logistic
    'multiclass_c': 1.0,

    # inference tuning
    'topN_for_review': 100,
    'initial_confidence_threshold': 0.90,  # high cutoff for binary 'advice' probability; will be calibrated
    'max_candidates_per_call': 10,

    # embedding batch size for inference (affects memory)
    'embed_batch_size': 512,
}

os.makedirs(CONFIG['model_dir'], exist_ok=True)

################################################################################
# Utilities: regex snippet extraction and augmentation
################################################################################
# compile regex for candidates
ADVICE_PATTERN = re.compile('|'.join(CONFIG['advice_keywords']), flags=re.IGNORECASE)

NEG_PATTERNS = [re.compile(p, flags=re.IGNORECASE) for p in CONFIG['negation_words']]

# improved snippet extractor: capture X words before and Y words after the matched token
def extract_snippets_from_text(text: str, left_words=None, right_words=None, max_snippet_words=None) -> List[str]:
    if left_words is None:
        left_words = CONFIG['left_words']
    if right_words is None:
        right_words = CONFIG['right_words']
    if max_snippet_words is None:
        max_snippet_words = CONFIG['max_snippet_words']

    snippets = []
    # Tokenize by whitespace to count words
    words = text.split()
    # we will search using regex on text but then determine word index
    for m in ADVICE_PATTERN.finditer(text):
        # char span
        start_c, end_c = m.span()
        # find word index of match start by counting characters before match
        char_before = text[:start_c]
        left_count = len(char_before.split())
        # compute snippet word indices
        start_idx = max(0, left_count - left_words)
        # how many words matched? get end word index similarly
        char_after = text[:end_c]
        end_count = len(char_after.split())
        end_idx = min(len(words), end_count + right_words)
        # cap size
        if (end_idx - start_idx) > max_snippet_words:
            end_idx = start_idx + max_snippet_words
        snippet = ' '.join(words[start_idx:end_idx])
        snippets.append(snippet)
    # deduplicate near-duplicates (keep order)
    seen = set()
    out = []
    for s in snippets:
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

# heuristic augmentation: produce negated counterexamples for ambiguous phrases
def generate_negation_examples(df: pd.DataFrame, passage_col: str='passage', label_col: str='type', n_per_class: int=500) -> pd.DataFrame:
    """
    For passages containing advice keyword but also negation words,
    treat them as negative examples (no_advice). Also create synthetic negation versions
    of positive examples by inserting explicit negation phrases - use sparingly.
    """
    rows = []
    # find candidate examples that contain advice keywords and negation words - treat as no advice
    for idx, r in df.iterrows():
        p = str(r[passage_col])
        if ADVICE_PATTERN.search(p) and any(pat.search(p) for pat in NEG_PATTERNS):
            rows.append({passage_col: p, label_col: 0})
    # also generate synthetic negations from positive samples (insert "I can't recommend" clause)
    positives = df[df[label_col] != 0].sample(min(len(df), n_per_class), random_state=CONFIG['random_state'])
    for _, r in positives.iterrows():
        p = r[passage_col]
        # insert a negation at start or after first sentence
        insert_phrases = ["I can't recommend that.", "I'm not able to recommend that.", "We cannot recommend doing that right now."]
        ins = random.choice(insert_phrases)
        # Insert after first punctuation (or at start)
        if '.' in p:
            parts = p.split('.', 1)
            new_p = parts[0] + '. ' + ins + ' ' + (parts[1].strip() if len(parts) > 1 else '')
        else:
            new_p = ins + ' ' + p
        rows.append({passage_col: new_p, label_col: 0})
    out_df = pd.DataFrame(rows)
    return out_df

################################################################################
# Training pipeline
################################################################################
def load_and_prepare_training(csv_path: str):
    df = pd.read_csv(csv_path)
    # normalize column names if needed
    if 'passage' not in df.columns:
        raise ValueError("CSV missing 'passage' column")
    if 'type' not in df.columns:
        raise ValueError("CSV missing 'type' column")
    df = df.dropna(subset=['passage', 'type'])
    # lowercase + remove repeated whitespace
    df['passage'] = df['passage'].astype(str).apply(lambda x: ' '.join(x.split()))
    df['type'] = df['type'].astype(int)
    return df

def augment_training(df: pd.DataFrame, augment_negations: bool=True) -> pd.DataFrame:
    out = df.copy()
    if augment_negations:
        negs = generate_negation_examples(df)
        if len(negs) > 0:
            out = pd.concat([out, negs], ignore_index=True)
    # Optional: can add other augmentations here (synonym swaps, back-translation etc.)
    return out.sample(frac=1.0, random_state=CONFIG['random_state']).reset_index(drop=True)

def train_encoder(save_path: str):
    print("Loading sentence-transformer encoder:", CONFIG['encoder_model_name'])
    encoder = SentenceTransformer(CONFIG['encoder_model_name'])
    # Save the encoder using joblib or encoder.save() - prefer built-in .save for ST models
    enc_save = Path(save_path)
    encoder.save(str(enc_save))
    return encoder

def load_encoder(save_path: str) -> SentenceTransformer:
    # SentenceTransformer expects directory path
    encoder = SentenceTransformer(save_path)
    return encoder

def encode_texts(encoder: SentenceTransformer, texts: List[str], batch_size: int=CONFIG['embed_batch_size']) -> np.ndarray:
    embeddings = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    return embeddings

def train_classifiers(df: pd.DataFrame, encoder: SentenceTransformer, model_dir: str=CONFIG['model_dir']):
    """
    Trains:
      - Binary classifier: 0 vs >0
      - Multiclass classifier: 1/2/3 (trained only on positive labeled samples)
    Saves models to disk.
    """
    df = df.copy()
    df['is_advice'] = (df['type'] != 0).astype(int)

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=df['is_advice'])
    # encode
    print("Encoding training set passages...")
    X_train = encode_texts(encoder, train_df['passage'].tolist())
    X_test = encode_texts(encoder, test_df['passage'].tolist())

    y_train_bin = train_df['is_advice'].values
    y_test_bin = test_df['is_advice'].values

    # Binary clf
    print("Training binary classifier (advice vs not)...")
    if CONFIG['binary_clf_type'] == 'lgb' and HAS_LGB:
        dtrain = lgb.Dataset(X_train, label=y_train_bin)
        params = {'objective':'binary', 'metric':'binary_logloss', 'verbosity': -1}
        bst = lgb.train(params, dtrain, num_boost_round=200)
        binary_clf = bst
        dump(binary_clf, os.path.join(model_dir, CONFIG['binary_clf_name']))
    else:
        # Use logistic regression with class weighting and calibration for better probabilities
        clf = LogisticRegression(C=CONFIG['binary_c'], max_iter=1000, class_weight='balanced', solver='lbfgs')
        # Calibrate to get better probability estimates (important for thresholding)
        cal = CalibratedClassifierCV(base_estimator=clf, cv=3)
        cal.fit(X_train, y_train_bin)
        binary_clf = cal
        dump(binary_clf, os.path.join(model_dir, CONFIG['binary_clf_name']))
    print("Saved binary classifier to", os.path.join(model_dir, CONFIG['binary_clf_name']))

    # Evaluate and find threshold for top-N precision
    y_probs = binary_clf.predict_proba(X_test)[:, 1]
    # compute top-N precision on test set
    def precision_at_N(y_true, probs, N=100):
        # N relative to test set size
        idx = np.argsort(probs)[::-1][:N]
        return precision_score(y_true[idx], (probs[idx] > 0.5).astype(int)) if len(idx) > 0 else 0.0

    # Use precision-recall curve to pick a threshold achieving high precision
    prec, rec, thresh = precision_recall_curve(y_test_bin, y_probs)
    # find thresholds with precision >= 0.95 (or target) and maximize recall there
    target_prec = 0.95
    valid = np.where(prec[:-1] >= target_prec)[0]
    if len(valid) > 0:
        chosen = valid[np.argmax(rec[:-1][valid])]
        chosen_threshold = thresh[chosen]
    else:
        # fallback to a high percentile
        chosen_threshold = np.percentile(y_probs, 99)
    print(f"Chosen binary probability threshold for ~{target_prec*100}% precision on test set: {chosen_threshold:.4f}")

    # Save threshold and evaluation metrics
    metrics = {
        'chosen_binary_threshold': float(chosen_threshold),
        'test_accuracy': float(accuracy_score(y_test_bin, (y_probs >= 0.5).astype(int))),
        'test_precision': float(precision_score(y_test_bin, (y_probs >= 0.5).astype(int))),
    }
    with open(os.path.join(model_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Multiclass classifier trained only on positive examples
    pos_train = train_df[train_df['is_advice'] == 1]
    pos_test = test_df[test_df['is_advice'] == 1]
    if len(pos_train) < 10:
        print("Not enough positive examples for multiclass training; skipping multiclass stage")
        return binary_clf, None

    print("Encoding positive subset for multiclass classification...")
    X_pos_train = encode_texts(encoder, pos_train['passage'].tolist())
    X_pos_test = encode_texts(encoder, pos_test['passage'].tolist())
    y_pos_train = pos_train['type'].values
    y_pos_test = pos_test['type'].values

    # Use LogisticRegression (multinomial) or LightGBM multiclass
    if HAS_LGB:
        lgb_train = lgb.Dataset(X_pos_train, label=y_pos_train)
        params = {'objective':'multiclass', 'num_class': len(np.unique(y_pos_train)), 'metric':'multi_logloss', 'verbosity': -1}
        bst = lgb.train(params, lgb_train, num_boost_round=200)
        multiclass_clf = bst
        dump(multiclass_clf, os.path.join(model_dir, CONFIG['multiclass_clf_name']))
    else:
        mc = LogisticRegression(C=CONFIG['multiclass_c'], max_iter=1000, multi_class='multinomial', solver='lbfgs')
        mc.fit(X_pos_train, y_pos_train)
        multiclass_clf = mc
        dump(multiclass_clf, os.path.join(model_dir, CONFIG['multiclass_clf_name']))

    print("Saved multiclass classifier to", os.path.join(model_dir, CONFIG['multiclass_clf_name']))

    return binary_clf, multiclass_clf

################################################################################
# Inference pipeline (apply to transcripts folder)
################################################################################
def extract_candidate_snippets_from_transcript_file(path: Path, max_candidates: int=CONFIG['max_candidates_per_call']) -> List[Tuple[str, str]]:
    """
    Given a transcript file (assumed plain text), extract candidate snippets from agent speech.
    We assume transcripts are pre-processed to separate agent/customer. If not, pass full transcript.
    Returns list of (snippet_text, metadata) where metadata may include snippet location index.
    """
    text = path.read_text(encoding='utf-8', errors='ignore')
    # if your transcripts separate speaker labels, prefer to extract agent-only speech; else pass full text
    # Simple heuristic: if labels like "Agent:" or "Rep:" present, strip other speaker lines
    # Otherwise process full text
    lines = text.splitlines()
    agent_lines = []
    has_labels = any(re.match(r'^(Agent|Rep|Representative|Advisor)\s*[:\-]', ln, flags=re.IGNORECASE) for ln in lines)
    if has_labels:
        for ln in lines:
            if re.match(r'^(Agent|Rep|Representative|Advisor)\s*[:\-]', ln, flags=re.IGNORECASE):
                # remove leading label
                content = re.sub(r'^(Agent|Rep|Representative|Advisor)\s*[:\-]\s*', '', ln, flags=re.IGNORECASE)
                agent_lines.append(content)
        agent_text = ' '.join(agent_lines)
    else:
        agent_text = text

    # extract snippets
    snippets = extract_snippets_from_text(agent_text)
    return [(s, f"{path.name}:{i}") for i, s in enumerate(snippets[:max_candidates])]

def inference_on_transcripts(transcript_dir: str, encoder: SentenceTransformer, binary_clf, multiclass_clf=None, out_csv: str='results_topN.csv'):
    transcript_dir = Path(transcript_dir)
    rows_out = []
    # process files
    all_candidates_texts = []
    all_candidates_meta = []
    all_file_map = []  # map candidate idx to source file
    print("Scanning transcripts and extracting candidates...")
    for path in tqdm(list(transcript_dir.glob('**/*.txt'))):
        try:
            cands = extract_candidate_snippets_from_transcript_file(path)
            for snippet, meta in cands:
                all_candidates_texts.append(snippet)
                all_candidates_meta.append({'meta': meta, 'file': path.name})
        except Exception as e:
            print("Error reading", path, e)
    if len(all_candidates_texts) == 0:
        print("No candidate snippets found in transcript dir.")
        return

    print("Encoding all {} candidate snippets...".format(len(all_candidates_texts)))
    X = encode_texts(encoder, all_candidates_texts)

    # binary scoring
    if hasattr(binary_clf, 'predict_proba'):
        probs = binary_clf.predict_proba(X)[:, 1]
    else:
        # lightgbm case
        probs = binary_clf.predict(X)

    # apply chosen threshold from metrics if exists
    metrics_path = os.path.join(CONFIG['model_dir'], 'training_metrics.json')
    chosen_threshold = CONFIG['initial_confidence_threshold']
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            if 'chosen_binary_threshold' in metrics:
                chosen_threshold = metrics['chosen_binary_threshold']

    # Build result rows with binary score
    result_rows = []
    for i, meta in enumerate(all_candidates_meta):
        result_rows.append({
            'file': meta['file'],
            'meta': meta['meta'],
            'snippet': all_candidates_texts[i],
            'binary_prob': float(probs[i])
        })
    # rank by binary_prob descending
    df_res = pd.DataFrame(result_rows).sort_values('binary_prob', ascending=False).reset_index(drop=True)

    # apply threshold to get positives for multiclass
    positives = df_res[df_res['binary_prob'] >= chosen_threshold].copy()
    print(f"Found {len(positives)} positives above threshold {chosen_threshold:.4f}")

    if multiclass_clf is not None and len(positives) > 0:
        # encode positives snippets (already encoded as X; we need indices)
        pos_indices = positives.index.tolist()
        X_pos = X[pos_indices]
        # predict multiclass
        if hasattr(multiclass_clf, 'predict_proba'):
            mc_probs = multiclass_clf.predict_proba(X_pos)
            mc_preds = np.argmax(mc_probs, axis=1)
            mc_scores = np.max(mc_probs, axis=1)
        else:
            mc_preds = multiclass_clf.predict(X_pos)
            mc_scores = np.ones_like(mc_preds, dtype=float)  # no probability
        # attach
        positives['mc_pred'] = mc_preds
        positives['mc_score'] = mc_scores
        positives['mc_label'] = positives['mc_pred'].map(CONFIG['label_map'])
    else:
        positives['mc_pred'] = None
        positives['mc_score'] = None
        positives['mc_label'] = None

    # Output top-N for review
    topN = CONFIG['topN_for_review']
    top_df = positives.sort_values('binary_prob', ascending=False).head(topN)
    top_df.to_csv(out_csv, index=False)
    print(f"Wrote top-{topN} candidate snippets to {out_csv}")

    return df_res, top_df

################################################################################
# CLI driver
################################################################################
def main(args):
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    if args.train:
        # load and augment
        src = args.train_csv or 'training_data/model_cases7.csv'
        print("Loading training data from", src)
        df = load_and_prepare_training(src)
        df_aug = augment_training(df, augment_negations=True)
        print("Dataset size after augmentation:", len(df_aug))
        # train encoder
        encoder_path = os.path.join(CONFIG['model_dir'], 'encoder')
        if not os.path.exists(encoder_path):
            encoder = train_encoder(encoder_path)
        else:
            print("Encoder already exists at", encoder_path, "- loading")
            encoder = load_encoder(encoder_path)
        # train classifiers
        train_classifiers(df_aug, encoder, model_dir=CONFIG['model_dir'])
        print("Training complete. Models in", CONFIG['model_dir'])

    if args.infer:
        # load models
        encoder_path = os.path.join(CONFIG['model_dir'], 'encoder')
        if not os.path.exists(encoder_path):
            raise RuntimeError("Encoder not found; train first.")
        encoder = load_encoder(encoder_path)
        binary_clf = load(os.path.join(CONFIG['model_dir'], CONFIG['binary_clf_name']))
        multiclass_clf_path = os.path.join(CONFIG['model_dir'], CONFIG['multiclass_clf_name'])
        multiclass_clf = None
        if os.path.exists(multiclass_clf_path):
            multiclass_clf = load(multiclass_clf_path)
        transcript_dir = args.transcript_dir or '.'
        out_csv = args.out or 'results_topN.csv'
        inference_on_transcripts(transcript_dir, encoder, binary_clf, multiclass_clf, out_csv=out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train models from CSV')
    parser.add_argument('--train-csv', dest='train_csv', default=None, help='Path to training CSV')
    parser.add_argument('--infer', action='store_true', help='Run inference on transcripts')
    parser.add_argument('--transcript-dir', dest='transcript_dir', default=None, help='Directory with transcripts (.txt)')
    parser.add_argument('--out', dest='out', default='results_topN.csv', help='Output CSV top N')
    args = parser.parse_args()
    main(args)
