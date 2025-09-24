class ContrastiveLearningTrainer:
    
    def train_setfit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Train SetFit model and save as checkpoint
        """
        # ... training code ...
        
        # After training, save as checkpoint instead of pickle
        checkpoint_path = f"{self.config.model_save_path}/setfit_checkpoint"
        self.model.save_pretrained(checkpoint_path)
        logger.info(f"SetFit model saved to checkpoint: {checkpoint_path}")
        
        return self.model

class FinancialAdviceDetectionPipeline:
    
    def load_trained_models(self):
        """
        Updated to load SetFit from checkpoint
        """
        # ... other model loading ...
        
        # Load SetFit from checkpoint
        setfit_checkpoint = f"{self.config.model_save_path}/setfit_checkpoint"
        if os.path.exists(setfit_checkpoint):
            logger.info(f"Loading SetFit model from {setfit_checkpoint}")
            try:
                self.setfit_model = SetFitModel.from_pretrained(setfit_checkpoint)
                logger.info("✓ SetFit model loaded from checkpoint")
            except Exception as e:
                logger.error(f"Failed to load SetFit checkpoint: {e}")
        else:
            logger.warning(f"SetFit checkpoint not found at {setfit_checkpoint}")
