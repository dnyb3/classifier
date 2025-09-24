class FinancialAdviceDetectionPipeline:
    
    def _process_single_transcript(self, interaction_id: str, transcript: str) -> Dict:
        """
        FIXED: Process transcript through ALL pipeline stages
        """
        result = {
            'interaction_id': interaction_id,
            'predicted_label': 0,
            'confidence_score': 0.0,
            'extracted_snippet': '',
            'stage_passed': 'rule_filter',
            'review_priority': 'low',
            'ensemble_predictions': {},
            'model_agreement': False
        }
        
        try:
            # ============================================================
            # STAGE 1: Rule-based Pre-filtering (High Recall)
            # ============================================================
            should_process, segments = self.rule_filter.filter_transcript(transcript)
            
            if not should_process:
                return result  # ~70-85% filtered here
            
            result['stage_passed'] = 'context_extraction'
            
            # ============================================================
            # STAGE 2: Smart Context Extraction
            # ============================================================
            contexts = []
            for segment in segments:
                semantic_contexts = self.context_extractor.extract_semantic_context(
                    transcript, 
                    [segment['start_char']]
                )
                contexts.extend(semantic_contexts)
            
            if not contexts:
                return result
            
            result['stage_passed'] = 'ml_classification'
            
            # ============================================================
            # STAGE 3: Multi-Model Classification
            # ============================================================
            all_predictions = []
            
            # 3a. Neural Model (BERT/MobileBERT) Predictions
            neural_predictions = self._get_neural_predictions(contexts)
            all_predictions.append(('neural', neural_predictions))
            
            # 3b. SetFit/Embedding Model Predictions
            if self.setfit_model is not None:
                setfit_predictions = self._get_setfit_predictions(contexts)
                all_predictions.append(('setfit', setfit_predictions))
            
            # 3c. Ensemble Model Predictions (using embeddings)
            if self.ensemble_model is not None:
                ensemble_predictions = self._get_ensemble_predictions(contexts)
                all_predictions.append(('ensemble', ensemble_predictions))
            
            # ============================================================
            # STAGE 4: Aggregate Predictions (Voting/Stacking)
            # ============================================================
            final_prediction = self._aggregate_predictions(all_predictions)
            
            result['predicted_label'] = final_prediction['label']
            result['confidence_score'] = final_prediction['confidence']
            result['extracted_snippet'] = final_prediction['snippet'][:500]
            result['ensemble_predictions'] = final_prediction['individual_scores']
            result['model_agreement'] = final_prediction['agreement']
            
            # ============================================================
            # STAGE 5: Confidence-Based Prioritization
            # ============================================================
            if result['model_agreement'] and result['confidence_score'] >= self.config.confidence_threshold:
                result['stage_passed'] = 'high_confidence'
                result['review_priority'] = 'high'
            elif result['confidence_score'] >= self.config.confidence_threshold:
                result['stage_passed'] = 'medium_confidence'
                result['review_priority'] = 'medium'
            elif not result['model_agreement']:
                result['review_priority'] = 'conflicted'  # Models disagree
            
        except Exception as e:
            logger.error(f"Error processing transcript {interaction_id}: {str(e)}")
            
        return result
    
    def _get_neural_predictions(self, contexts: List[str]) -> List[Dict]:
        """
        Get predictions from neural model (BERT/ONNX)
        """
        predictions = []
        
        for context in contexts:
            if self.config.use_onnx and self.onnx_session:
                try:
                    logits = self.onnx_optimizer.batch_inference(
                        [context], 
                        self.neural_trainer.tokenizer
                    )
                    proba = self._softmax(logits[0])
                except:
                    proba = self._pytorch_inference(context)
            else:
                proba = self._pytorch_inference(context)
            
            predictions.append({
                'text': context,
                'label': int(np.argmax(proba)),
                'confidence': float(np.max(proba)),
                'probabilities': proba.tolist()
            })
        
        return predictions
    
    def _get_setfit_predictions(self, contexts: List[str]) -> List[Dict]:
        """
        Get predictions from SetFit/Embedding model
        """
        predictions = []
        
        if hasattr(self.setfit_model, 'predict_proba'):
            probas = self.setfit_model.predict_proba(contexts)
            labels = self.setfit_model.predict(contexts)
            
            for i, context in enumerate(contexts):
                predictions.append({
                    'text': context,
                    'label': int(labels[i]),
                    'confidence': float(np.max(probas[i])),
                    'probabilities': probas[i].tolist()
                })
        
        return predictions
    
    def _get_ensemble_predictions(self, contexts: List[str]) -> List[Dict]:
        """
        Get predictions from ensemble classifier
        """
        predictions = []
        
        # Generate embeddings for contexts
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode(contexts, normalize_embeddings=True)
        
        # Get ensemble predictions
        if self.ensemble.calibrator is not None:
            labels = self.ensemble.calibrator.predict(embeddings)
            probas = self.ensemble.calibrator.predict_proba(embeddings)
            
            for i, context in enumerate(contexts):
                predictions.append({
                    'text': context,
                    'label': int(labels[i]),
                    'confidence': float(np.max(probas[i])),
                    'probabilities': probas[i].tolist()
                })
        
        return predictions
    
    def _aggregate_predictions(self, all_predictions: List[Tuple[str, List[Dict]]]) -> Dict:
        """
        Aggregate predictions from multiple models using weighted voting
        """
        if not all_predictions:
            return {
                'label': 0,
                'confidence': 0.0,
                'snippet': '',
                'individual_scores': {},
                'agreement': False
            }
        
        # Weight configuration based on expected performance
        model_weights = {
            'neural': 0.4,    # Primary model
            'ensemble': 0.35, # High precision ensemble
            'setfit': 0.25    # Contrastive learning model
        }
        
        # Find best prediction across all models and contexts
        best_overall = {
            'confidence': 0.0,
            'label': 0,
            'text': '',
            'weighted_score': 0.0
        }
        
        all_model_predictions = {}
        
        for model_name, predictions in all_predictions:
            if not predictions:
                continue
                
            # Get best prediction from this model
            best_pred = max(predictions, key=lambda x: x['confidence'])
            all_model_predictions[model_name] = best_pred
            
            # Calculate weighted score
            weight = model_weights.get(model_name, 0.33)
            weighted_confidence = best_pred['confidence'] * weight
            
            # Track if this is the best overall
            if weighted_confidence > best_overall['weighted_score']:
                best_overall = {
                    'confidence': best_pred['confidence'],
                    'label': best_pred['label'],
                    'text': best_pred['text'],
                    'weighted_score': weighted_confidence,
                    'probabilities': best_pred['probabilities']
                }
        
        # Check model agreement (for high confidence)
        if len(all_model_predictions) > 1:
            labels = [p['label'] for p in all_model_predictions.values()]
            agreement = len(set(labels)) == 1
        else:
            agreement = True
        
        # Boost confidence if models agree
        final_confidence = best_overall['confidence']
        if agreement and len(all_model_predictions) > 1:
            final_confidence = min(1.0, final_confidence * 1.1)  # 10% boost for agreement
        
        return {
            'label': best_overall['label'],
            'confidence': final_confidence,
            'snippet': best_overall['text'],
            'individual_scores': {
                model: {
                    'label': pred['label'],
                    'confidence': pred['confidence']
                }
                for model, pred in all_model_predictions.items()
            },
            'agreement': agreement
        }
    
    def _extract_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        FIXED: Extract features for ensemble training
        Also store the embedder for later use in inference
        """
        # Store embedder for inference
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        X_train = self.embedder.encode(train_df['passage'].tolist(), show_progress_bar=True)
        X_val = self.embedder.encode(val_df['passage'].tolist(), show_progress_bar=True)
        
        return X_train, X_val
    
    def _create_output_dataframe(self, original_df: pd.DataFrame, results: List[Dict]) -> pd.DataFrame:
        """
        ENHANCED: Create output with detailed model predictions
        """
        output_df = original_df.copy()
        
        # Add prediction columns
        output_df['PREDICTED_LABEL'] = [r['predicted_label'] for r in results]
        output_df['CONFIDENCE_SCORE'] = [r['confidence_score'] for r in results]
        output_df['EXTRACTED_SNIPPET'] = [r['extracted_snippet'] for r in results]
        output_df['REVIEW_PRIORITY'] = [r['review_priority'] for r in results]
        output_df['PIPELINE_STAGE'] = [r['stage_passed'] for r in results]
        output_df['MODEL_AGREEMENT'] = [r['model_agreement'] for r in results]
        
        # Add individual model scores for analysis
        for i, r in enumerate(results):
            for model_name, scores in r.get('ensemble_predictions', {}).items():
                output_df.loc[i, f'{model_name.upper()}_LABEL'] = scores['label']
                output_df.loc[i, f'{model_name.upper()}_CONFIDENCE'] = scores['confidence']
        
        # Map labels to readable names
        label_map = {
            0: 'No Advice',
            1: 'IRA Rollover',
            2: 'Stay in Plan',
            3: 'Plan to Plan Transfer'
        }
        output_df['PREDICTED_LABEL_NAME'] = output_df['PREDICTED_LABEL'].map(label_map)
        
        # Sort by confidence score descending
        output_df = output_df.sort_values('CONFIDENCE_SCORE', ascending=False)
        
        return output_df
