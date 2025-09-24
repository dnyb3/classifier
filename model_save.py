class FinancialAdviceDetectionPipeline:
    
    def load_trained_models(self):
        """
        Load all trained models from disk for inference/evaluation without retraining
        """
        logger.info("=" * 80)
        logger.info("LOADING TRAINED MODELS FROM DISK")
        logger.info("=" * 80)
        
        try:
            # Load neural model (BERT/MobileBERT)
            model_path = f"{self.config.model_save_path}/best_model"
            if os.path.exists(model_path):
                logger.info(f"Loading neural model from {model_path}")
                self.neural_model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.neural_trainer.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.neural_model.eval()
                logger.info("✓ Neural model loaded successfully")
            else:
                logger.warning(f"Neural model not found at {model_path}")
            
            # Load ONNX session if enabled
            if self.config.use_onnx:
                onnx_path = f"{self.config.model_save_path}/model_{self.config.quantization_type}.onnx"
                if not os.path.exists(onnx_path):
                    # Try without quantization suffix
                    onnx_path = f"{self.config.model_save_path}/model.onnx"
                
                if os.path.exists(onnx_path):
                    logger.info(f"Loading ONNX model from {onnx_path}")
                    self.onnx_session = self.onnx_optimizer.create_inference_session(onnx_path)
                    logger.info("✓ ONNX session created successfully")
                else:
                    logger.warning(f"ONNX model not found at {onnx_path}")
                    self.config.use_onnx = False
            
            # Load ensemble models (if saved)
            ensemble_path = f"{self.config.model_save_path}/ensemble_models.pkl"
            if os.path.exists(ensemble_path):
                logger.info(f"Loading ensemble models from {ensemble_path}")
                import pickle
                with open(ensemble_path, 'rb') as f:
                    ensemble_data = pickle.load(f)
                    self.ensemble.models = ensemble_data['models']
                    self.ensemble.ensemble = ensemble_data['ensemble']
                    self.ensemble.calibrator = ensemble_data['calibrator']
                logger.info("✓ Ensemble models loaded successfully")
            else:
                logger.warning(f"Ensemble models not found at {ensemble_path}")
            
            # Load contrastive/embedding model
            embedding_model_path = f"{self.config.model_save_path}/embedding_model.pkl"
            if os.path.exists(embedding_model_path):
                logger.info(f"Loading embedding model from {embedding_model_path}")
                import pickle
                with open(embedding_model_path, 'rb') as f:
                    embedding_data = pickle.load(f)
                    self.setfit_model = embedding_data.get('model')
                    if hasattr(self.contrastive_trainer, 'encoder'):
                        self.contrastive_trainer.encoder = embedding_data.get('encoder')
                    if hasattr(self.contrastive_trainer, 'classifier'):
                        self.contrastive_trainer.classifier = embedding_data.get('classifier')
                    if hasattr(self.contrastive_trainer, 'scaler'):
                        self.contrastive_trainer.scaler = embedding_data.get('scaler')
                    if hasattr(self.contrastive_trainer, 'centroids'):
                        self.contrastive_trainer.centroids = embedding_data.get('centroids')
                logger.info("✓ Embedding model loaded successfully")
            else:
                logger.warning(f"Embedding model not found at {embedding_model_path}")
            
            logger.info("=" * 80)
            logger.info("MODEL LOADING COMPLETE")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def save_trained_models(self):
        """
        Save all trained models to disk for later use
        """
        logger.info("Saving trained models...")
        
        # Ensure directory exists
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
        
        # Save ensemble models
        if self.ensemble.calibrator is not None:
            ensemble_path = f"{self.config.model_save_path}/ensemble_models.pkl"
            import pickle
            with open(ensemble_path, 'wb') as f:
                pickle.dump({
                    'models': self.ensemble.models,
                    'ensemble': self.ensemble.ensemble,
                    'calibrator': self.ensemble.calibrator
                }, f)
            logger.info(f"✓ Ensemble models saved to {ensemble_path}")
        
        # Save embedding/contrastive model
        if self.setfit_model is not None or hasattr(self.contrastive_trainer, 'classifier'):
            embedding_model_path = f"{self.config.model_save_path}/embedding_model.pkl"
            import pickle
            with open(embedding_model_path, 'wb') as f:
                pickle.dump({
                    'model': self.setfit_model,
                    'encoder': getattr(self.contrastive_trainer, 'encoder', None),
                    'classifier': getattr(self.contrastive_trainer, 'classifier', None),
                    'scaler': getattr(self.contrastive_trainer, 'scaler', None),
                    'centroids': getattr(self.contrastive_trainer, 'centroids', None)
                }, f)
            logger.info(f"✓ Embedding model saved to {embedding_model_path}")
        
        logger.info("Model saving complete")
    
    def evaluate_only(self, test_data_path: str):
        """
        Run evaluation only using pre-trained models
        """
        logger.info("=" * 80)
        logger.info("EVALUATION MODE - LOADING PRE-TRAINED MODELS")
        logger.info("=" * 80)
        
        # Load models from disk
        if not self.load_trained_models():
            logger.error("Failed to load models. Please train the pipeline first.")
            return
        
        # Load test data
        logger.info(f"Loading test data from {test_data_path}")
        test_df = pd.read_csv(test_data_path)
        
        # Clean test data
        test_df['passage'] = test_df['passage'].str.lower()
        test_df['passage'] = test_df['passage'].apply(lambda x: ' '.join(x.split()))
        test_df = test_df.dropna(subset=['passage', 'type'])
        test_df['type'] = test_df['type'].astype(int)
        
        logger.info(f"Test data loaded: {len(test_df)} samples")
        
        # Run evaluation
        self._evaluate_pipeline(test_df)
    
    def train_pipeline(self, training_data_path: str):
        """
        Train the complete pipeline (updated to save models)
        """
        logger.info("=" * 80)
        logger.info("STARTING PIPELINE TRAINING")
        logger.info("=" * 80)
        
        # Load and prepare data
        logger.info("Loading training data...")
        train_df, val_df, test_df = self._prepare_data(training_data_path)
        
        # Stage 1: Data augmentation
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 1: Data Augmentation")
        logger.info("=" * 40)
        augmented_train_df = self.augmentor.augment_dataset(train_df)
        
        # Stage 2: Train neural model (MobileBERT)
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 2: Neural Model Training")
        logger.info("=" * 40)
        self.neural_model = self.neural_trainer.train_model(augmented_train_df, val_df)
        
        # Stage 3: Train SetFit with contrastive learning
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 3: Contrastive Learning")
        logger.info("=" * 40)
        self.setfit_model = self.contrastive_trainer.train_setfit(train_df, val_df)
        
        # Stage 4: Extract features and train ensemble
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 4: Ensemble Training")
        logger.info("=" * 40)
        X_train, X_val = self._extract_features(augmented_train_df, val_df)
        self.ensemble_model = self.ensemble.build_ensemble(
            X_train, augmented_train_df['type'].values,
            X_val, val_df['type'].values
        )
        
        # Stage 5: Export and optimize for production
        if self.config.use_onnx:
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 5: ONNX Optimization")
            logger.info("=" * 40)
            
            # Export to ONNX
            onnx_path = self.neural_trainer.export_to_onnx(
                f"{self.config.model_save_path}/best_model",
                f"{self.config.model_save_path}/model.onnx"
            )
            
            # Optimize
            optimized_path = self.onnx_optimizer.optimize_model(onnx_path)
            self.onnx_session = self.onnx_optimizer.create_inference_session(optimized_path)
        
        # Save all models for later use
        logger.info("\n" + "=" * 40)
        logger.info("SAVING TRAINED MODELS")
        logger.info("=" * 40)
        self.save_trained_models()
        
        # Final evaluation
        logger.info("\n" + "=" * 40)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 40)
        self._evaluate_pipeline(test_df)
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE TRAINING COMPLETE")
        logger.info("=" * 80)
    
    def _evaluate_pipeline(self, test_df: pd.DataFrame):
        """
        Evaluate the complete pipeline (with model loading check)
        """
        # Check if models are loaded
        if self.neural_model is None and self.onnx_session is None:
            logger.warning("No models loaded. Attempting to load from disk...")
            if not self.load_trained_models():
                logger.error("Cannot evaluate without trained models")
                return
        
        # Rest of the original evaluation code...
        predictions = []
        true_labels = test_df['type'].values
        
        for idx, row in test_df.iterrows():
            result = self._process_single_transcript(
                f"test_{idx}",
                row['passage']
            )
            predictions.append(result['predicted_label'])
        
        # Calculate metrics
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Calculate metrics for advice classes only (labels 1, 2, 3)
        advice_mask = true_labels > 0
        advice_predictions = [p for p, t in zip(predictions, true_labels) if t > 0]
        advice_true = true_labels[advice_mask]
        
        if len(advice_true) > 0:
            advice_precision = precision_score(advice_true, advice_predictions, average='weighted', zero_division=0)
            advice_recall = recall_score(advice_true, advice_predictions, average='weighted', zero_division=0)
        else:
            advice_precision = advice_recall = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        logger.info("\n" + "=" * 40)
        logger.info("PIPELINE EVALUATION RESULTS")
        logger.info("=" * 40)
        logger.info(f"Overall Precision: {precision:.3f}")
        logger.info(f"Overall Recall: {recall:.3f}")
        logger.info(f"Overall F1: {f1:.3f}")
        logger.info(f"\nAdvice Detection Precision: {advice_precision:.3f}")
        logger.info(f"Advice Detection Recall: {advice_recall:.3f}")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        self.metrics['test_precision'].append(precision)
        self.metrics['test_recall'].append(recall)
        self.metrics['test_f1'].append(f1)
        self.metrics['advice_precision'].append(advice_precision)
