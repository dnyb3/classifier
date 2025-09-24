def augment_dataset(self, train_df: pd.DataFrame) -> pd.DataFrame:
    """Augment training data with class balancing focus"""
    
    # Check current class distribution
    class_counts = train_df['type'].value_counts().to_dict()
    logger.info(f"Original distribution: {class_counts}")
    
    augmented_data = []
    
    # Strategy 1: Balance classes to reduce false positives
    if self.config.augmentation_strategy == "balanced":
        max_samples = max(class_counts.values())
        
        for label in range(4):
            current_count = class_counts.get(label, 0)
            samples_needed = max_samples - current_count
            
            if samples_needed > 0:
                label_data = train_df[train_df['type'] == label]
                
                # Generate enough samples to balance
                for _ in range(samples_needed):
                    synthetic = self._generate_synthetic_sample(label_data, label)
                    augmented_data.append(synthetic)
                    
                logger.info(f"Class {label}: Added {samples_needed} samples")
    
    # Strategy 2: Focus on minority advice classes
    elif self.config.augmentation_strategy == "focus_advice":
        # Only augment advice classes (1, 2, 3) heavily
        for label in [1, 2, 3]:
            label_data = train_df[train_df['type'] == label]
            samples_to_add = self.config.augmentation_ratios[label]
            
            for _ in range(samples_to_add):
                synthetic = self._generate_synthetic_sample(label_data, label)
                augmented_data.append(synthetic)
    
    augmented_df = pd.DataFrame(augmented_data)
    combined_df = pd.concat([train_df, augmented_df], ignore_index=True)
    
    logger.info(f"Final distribution: {combined_df['type'].value_counts().to_dict()}")
    logger.info(f"Augmented from {len(train_df)} to {len(combined_df)} samples")
    
    return combined_df

@dataclass
class PipelineConfig:
    # OPTION A: Aggressive augmentation for imbalanced classes
    use_augmentation: bool = True
    augmentation_strategy: str = "balanced"  # New parameter
    
    # Augment minority classes more heavily
    augmentation_ratios: dict = {
        0: 0,      # No advice - likely majority class, no augmentation
        1: 500,    # IRA advice - heavy augmentation 
        2: 500,    # Stay in plan - heavy augmentation
        3: 500     # Plan to plan - heavy augmentation
    }
    
    # OR OPTION B: Balance to match largest class
    target_samples_per_class: int = 1500  # Balance all classes to this
