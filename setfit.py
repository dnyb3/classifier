import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from setfit import SetFitModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_setfit_from_checkpoint(
    checkpoint_path: str = "optimized_models/setfit_checkpoint",
    test_data_path: str = "training_data/model_cases7.csv",
    test_split: float = 0.1
):
    """
    Evaluate SetFit model loaded from a Hugging Face checkpoint
    """
    logger.info("=" * 60)
    logger.info("SETFIT MODEL EVALUATION (FROM CHECKPOINT)")
    logger.info("=" * 60)
    
    # Load SetFit model from checkpoint
    logger.info(f"Loading SetFit model from {checkpoint_path}")
    try:
        # Load the SetFit model directly from the saved checkpoint
        model = SetFitModel.from_pretrained(checkpoint_path)
        logger.info("✓ Model loaded successfully")
        
        # Check model configuration
        if hasattr(model, 'model_body'):
            logger.info(f"Sentence transformer: {model.model_body.max_seq_length} max length")
        if hasattr(model, 'model_head'):
            if model.model_head is not None:
                logger.info(f"Classification head: {model.model_head}")
            else:
                logger.info("Using cosine similarity classifier")
                
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("\nTrying alternative loading methods...")
        
        try:
            # Alternative: Load from local directory
            from pathlib import Path
            if Path(checkpoint_path).exists():
                model = SetFitModel.from_pretrained(f"./{checkpoint_path}")
            else:
                logger.error(f"Checkpoint not found at {checkpoint_path}")
                return
        except Exception as e2:
            logger.error(f"Alternative loading also failed: {e2}")
            return
    
    # Load and prepare test data
    logger.info(f"\nLoading test data from {test_data_path}")
    df = pd.read_csv(test_data_path)
    
    # Clean data
    df['passage'] = df['passage'].str.lower()
    df['passage'] = df['passage'].apply(lambda x: ' '.join(x.split()))
    df = df.dropna(subset=['passage', 'type'])
    df['type'] = df['type'].astype(int)
    
    # Split off test set
    test_size = int(len(df) * test_split)
    test_df = df.tail(test_size)
    logger.info(f"Test set size: {len(test_df)} samples")
    
    # Get predictions
    test_texts = test_df['passage'].tolist()
    true_labels = test_df['type'].values
    
    logger.info("Generating predictions...")
    
    # Get predictions from SetFit model
    predictions = model.predict(test_texts)
    
    # Convert predictions to numpy array if needed
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    
    # Try to get probabilities (not all SetFit models support this)
    try:
        # SetFit models might have predict_proba
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(test_texts)
        else:
            # Try to get raw scores/distances
            logger.info("Model doesn't have predict_proba, trying to get raw scores...")
            embeddings = model.model_body.encode(test_texts)
            
            # If using a classification head
            if model.model_head is not None:
                import torch
                with torch.no_grad():
                    tensor_embeddings = torch.tensor(embeddings)
                    logits = model.model_head(tensor_embeddings)
                    probas = torch.softmax(logits, dim=-1).numpy()
            else:
                # Cosine similarity based - approximate probabilities
                probas = None
                logger.info("Probability scores not available for cosine similarity classifier")
    except Exception as e:
        logger.warning(f"Could not get probability scores: {e}")
        probas = None
    
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
    
    return {
        'model': model,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': predictions,
        'probabilities': probas
    }

def save_setfit_checkpoint(model, save_path: str = "optimized_models/setfit_checkpoint"):
    """
    Save a SetFit model to a checkpoint (for reference)
    """
    logger.info(f"Saving SetFit model to {save_path}")
    model.save_pretrained(save_path)
    logger.info("✓ Model saved successfully")

def load_and_inspect_checkpoint(checkpoint_path: str):
    """
    Load and inspect a SetFit checkpoint to understand its structure
    """
    import json
    from pathlib import Path
    
    checkpoint_dir = Path(checkpoint_path)
    
    logger.info(f"\nInspecting checkpoint at {checkpoint_path}")
    logger.info("-" * 40)
    
    # Check what files exist in the checkpoint
    if checkpoint_dir.exists():
        files = list(checkpoint_dir.iterdir())
        logger.info("Files in checkpoint:")
        for f in files:
            logger.info(f"  - {f.name}")
        
        # Check config file if it exists
        config_path = checkpoint_dir / "config_setfit.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("\nSetFit Configuration:")
            for key, value in config.items():
                logger.info(f"  {key}: {value}")
        
        # Check model card if it exists
        model_card_path = checkpoint_dir / "README.md"
        if model_card_path.exists():
            logger.info("\nModel card found")
        
        # Check for sentence transformer
        st_path = checkpoint_dir / "model_body"
        if st_path.exists():
            logger.info(f"\nSentence transformer found at {st_path}")
        
        # Check for classification head
        head_path = checkpoint_dir / "model_head.pkl"
        if head_path.exists():
            logger.info(f"Classification head found at {head_path}")
    else:
        logger.error(f"Checkpoint directory not found: {checkpoint_path}")

# Example usage
if __name__ == "__main__":
    # Method 1: Evaluate from checkpoint
    results = evaluate_setfit_from_checkpoint(
        checkpoint_path="optimized_models/setfit_checkpoint",
        test_data_path="training_data/model_cases7.csv",
        test_split=0.1
    )
    
    # Method 2: If you need to save a model to checkpoint format
    # (assuming you have a trained SetFit model)
    # save_setfit_checkpoint(your_model, "optimized_models/setfit_checkpoint")
    
    # Method 3: Inspect checkpoint structure
    # load_and_inspect_checkpoint("optimized_models/setfit_checkpoint")
