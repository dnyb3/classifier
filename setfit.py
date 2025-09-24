import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_setfit_model(model_path: str = "optimized_models/embedding_model.pkl", 
                          test_data_path: str = "training_data/model_cases7.csv",
                          test_split: float = 0.1):
    """
    Quick evaluation of SetFit/Embedding model
    """
    logger.info("=" * 60)
    logger.info("SETFIT MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Load the saved model
    logger.info(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            embedding_data = pickle.load(f)
            
        # Extract components based on what was saved
        setfit_model = embedding_data.get('model')
        encoder = embedding_data.get('encoder')
        classifier = embedding_data.get('classifier')
        scaler = embedding_data.get('scaler')
        centroids = embedding_data.get('centroids')
        
        if setfit_model is None and classifier is None:
            logger.error("No model found in the saved file!")
            return
            
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Load and prepare test data
    logger.info(f"Loading test data from {test_data_path}")
    df = pd.read_csv(test_data_path)
    
    # Clean data
    df['passage'] = df['passage'].str.lower()
    df['passage'] = df['passage'].apply(lambda x: ' '.join(x.split()))
    df = df.dropna(subset=['passage', 'type'])
    df['type'] = df['type'].astype(int)
    
    # Split off test set (last 10% by default)
    test_size = int(len(df) * test_split)
    test_df = df.tail(test_size)
    logger.info(f"Test set size: {len(test_df)} samples")
    
    # Get predictions
    test_texts = test_df['passage'].tolist()
    true_labels = test_df['type'].values
    
    logger.info("Generating predictions...")
    
    # Predict based on model type
    if setfit_model is not None:
        # SetFit model interface
        if hasattr(setfit_model, 'predict'):
            predictions = setfit_model.predict(test_texts)
            if hasattr(setfit_model, 'predict_proba'):
                probas = setfit_model.predict_proba(test_texts)
            else:
                probas = None
        else:
            logger.error("SetFit model doesn't have predict method")
            return
            
    elif classifier is not None and encoder is not None:
        # Fallback embedding + classifier interface
        logger.info("Using embedding + classifier approach")
        
        # Generate embeddings
        embeddings = encoder.encode(test_texts, show_progress_bar=True, normalize_embeddings=True)
        
        # Add centroid features if available
        if centroids is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            centroid_features = []
            for embedding in embeddings:
                features = []
                for label in sorted(centroids.keys()):
                    sim = cosine_similarity([embedding], [centroids[label]])[0][0]
                    features.append(sim)
                centroid_features.append(features)
            embeddings = np.hstack([embeddings, np.array(centroid_features)])
        
        # Scale if scaler available
        if scaler is not None:
            embeddings = scaler.transform(embeddings)
        
        # Predict
        predictions = classifier.predict(embeddings)
        probas = classifier.predict_proba(embeddings)
    else:
        logger.error("No valid model configuration found")
        return
    
    # Calculate metrics
    logger.info("\n" + "=" * 40)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 40)
    
    # Overall metrics
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    
    logger.info(f"Overall Precision: {precision:.3f}")
    logger.info(f"Overall Recall: {recall:.3f}")
    logger.info(f"Overall F1: {f1:.3f}")
    
    # Per-class metrics
    logger.info("\n" + "-" * 40)
    logger.info("PER-CLASS PERFORMANCE")
    logger.info("-" * 40)
    
    label_names = ['No Advice', 'IRA Rollover', 'Stay in Plan', 'Plan to Plan']
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=label_names,
        digits=3
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    logger.info("\nConfusion Matrix:")
    logger.info("Predicted →")
    logger.info("Actual ↓")
    print(cm)
    
    # High confidence analysis if probabilities available
    if probas is not None:
        logger.info("\n" + "-" * 40)
        logger.info("CONFIDENCE ANALYSIS")
        logger.info("-" * 40)
        
        max_probas = probas.max(axis=1)
        
        for threshold in [0.6, 0.7, 0.8, 0.9]:
            high_conf_mask = max_probas >= threshold
            n_high_conf = high_conf_mask.sum()
            
            if n_high_conf > 0:
                high_conf_precision = precision_score(
                    true_labels[high_conf_mask],
                    predictions[high_conf_mask],
                    average='weighted',
                    zero_division=0
                )
                coverage = n_high_conf / len(predictions) * 100
                logger.info(f"Threshold {threshold:.1f}: {n_high_conf} samples ({coverage:.1f}%), "
                          f"Precision: {high_conf_precision:.3f}")
    
    # Advice detection specific metrics (classes 1, 2, 3)
    logger.info("\n" + "-" * 40)
    logger.info("ADVICE DETECTION PERFORMANCE")
    logger.info("-" * 40)
    
    advice_mask = true_labels > 0
    if advice_mask.sum() > 0:
        advice_true = true_labels[advice_mask]
        advice_pred = predictions[advice_mask]
        
        # Binary classification: advice vs no-advice
        binary_true = (true_labels > 0).astype(int)
        binary_pred = (predictions > 0).astype(int)
        
        binary_precision = precision_score(binary_true, binary_pred, zero_division=0)
        binary_recall = recall_score(binary_true, binary_pred, zero_division=0)
        binary_f1 = f1_score(binary_true, binary_pred, zero_division=0)
        
        logger.info(f"Binary (Advice vs No-Advice):")
        logger.info(f"  Precision: {binary_precision:.3f}")
        logger.info(f"  Recall: {binary_recall:.3f}")
        logger.info(f"  F1: {binary_f1:.3f}")
        
        # Multi-class for advice types only
        if len(np.unique(advice_true)) > 1:
            advice_precision = precision_score(advice_true, advice_pred, average='weighted', zero_division=0)
            advice_recall = recall_score(advice_true, advice_pred, average='weighted', zero_division=0)
            logger.info(f"\nMulti-class (Advice Types):")
            logger.info(f"  Precision: {advice_precision:.3f}")
            logger.info(f"  Recall: {advice_recall:.3f}")
    
    # Sample predictions for inspection
    logger.info("\n" + "=" * 40)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("=" * 40)
    
    # Show 5 correct and 5 incorrect predictions
    correct_mask = predictions == true_labels
    incorrect_mask = ~correct_mask
    
    if probas is not None:
        confidence = probas.max(axis=1)
    else:
        confidence = np.ones(len(predictions))
    
    logger.info("\n✓ Sample CORRECT predictions:")
    correct_indices = np.where(correct_mask)[0][:5]
    for idx in correct_indices:
        logger.info(f"  True: {label_names[true_labels[idx]]}, "
                   f"Pred: {label_names[predictions[idx]]}, "
                   f"Conf: {confidence[idx]:.3f}")
        logger.info(f"  Text: {test_texts[idx][:100]}...")
    
    logger.info("\n✗ Sample INCORRECT predictions:")
    incorrect_indices = np.where(incorrect_mask)[0][:5]
    for idx in incorrect_indices:
        logger.info(f"  True: {label_names[true_labels[idx]]}, "
                   f"Pred: {label_names[predictions[idx]]}, "
                   f"Conf: {confidence[idx]:.3f}")
        logger.info(f"  Text: {test_texts[idx][:100]}...")
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': predictions,
        'probabilities': probas
    }

# Run the evaluation
if __name__ == "__main__":
    results = evaluate_setfit_model(
        model_path="optimized_models/embedding_model.pkl",
        test_data_path="training_data/model_cases7.csv",
        test_split=0.1  # Use last 10% as test
    )
