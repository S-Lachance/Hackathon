# Baseline Model Analysis

## Overview

This document provides a comprehensive understanding of the two baseline models provided in the RSN Hackathon Challenge for predicting Hospital One-year Mortality Risk (HOMR).

## The Challenge

**Objective**: Predict one-year mortality risk using admission data from healthcare records.

**Dataset**: Synthetic healthcare data with 248,487 patient visits and 246 features including:
- Demographics (age, gender, living status)
- Comorbidities (dx_* variables)
- Admission diagnoses (adm_* variables)
- Care utilization metrics (ED visits, ambulance usage)
- Current admission characteristics

## Baseline Models

### Model 1: Random Forest Trained on Synthetic Dataset

**Training Data**: Subset of the synthetic dataset

**Hyperparameters** (from `rf_homr_best_hps_synthetic.json`):
```json
{
    "n_estimators": 1024,
    "min_samples_leaf": 10,
    "max_features": 20,
    "class_weight": {
        "0": 0.827,
        "1": 0.173
    }
}
```

**Key Characteristics**:
- Uses 1024 decision trees (large ensemble for robustness)
- Class weight favors the negative class (no mortality) at ~82.7%
- Considers up to 20 features per split
- Minimum 10 samples per leaf node (prevents overfitting)

### Model 2: Random Forest Trained on Original Dataset

**Training Data**: Full original (real) dataset

**Hyperparameters** (from `rf_homr_best_hps_original.json`):
```json
{
    "n_estimators": 1024,
    "min_samples_leaf": 10,
    "max_features": 20,
    "class_weight": {
        "0": 0.742,
        "1": 0.258
    }
}
```

**Key Characteristics**:
- Same tree structure as Model 1 (1024 trees, 20 features, 10 samples/leaf)
- **Different class weighting**: More weight on positive class (mortality) at ~25.8%
- This suggests the original dataset may have different class distribution or requires different balancing

## Key Differences

### 1. **Class Weight Distribution**

The most significant difference between the two models is their class weighting:

| Model | Class 0 (No Mortality) | Class 1 (Mortality) |
|-------|------------------------|---------------------|
| Synthetic | 82.7% | 17.3% |
| Original | 74.2% | 25.8% |

**Implication**: Model 2 (Original) puts ~50% more weight on the positive class, suggesting:
- The original dataset may have a different class imbalance
- Real-world data may require more attention to catching true positives
- The synthetic data generation may have altered the class distribution

### 2. **Training Data Source**

- **Model 1**: Trained on synthetic data (generated to preserve privacy)
- **Model 2**: Trained on real patient data

**Key Questions**:
- How well does synthetic data capture real-world patterns?
- Does Model 1's performance on real test data match Model 2?
- Are there systematic biases introduced by synthetic data generation?

## Evaluation Strategy

To properly compare these models, we should evaluate on:

### Primary Metric: AUC-ROC
- **Why**: Handles class imbalance well
- **Target**: Higher is better (0.5 = random, 1.0 = perfect)
- **Clinical Relevance**: Measures discrimination ability across all thresholds

### Secondary Metrics:
1. **Precision**: Of predicted mortality cases, how many are correct?
2. **Recall (Sensitivity)**: Of actual mortality cases, how many do we catch?
3. **Specificity**: Of actual survivors, how many do we correctly identify?
4. **F1-Score**: Harmonic mean of precision and recall

### Important Considerations:
- **Clinical Cost**: False negatives (missing mortality risk) are likely more costly than false positives
- **Calibration**: Do predicted probabilities match actual risk?
- **Feature Importance**: Which features drive predictions?

## Expected Outcomes

### Hypothesis 1: Model 2 (Original) performs better
**Reasoning**: Trained on real data, should capture true patterns better

### Hypothesis 2: Model 1 (Synthetic) is competitive
**Reasoning**: If synthetic data generation is high-quality, performance should be similar

### Hypothesis 3: Different optimal operating points
**Reasoning**: Different class weights suggest different precision-recall trade-offs

## How to Evaluate

### Step 1: Download Pretrained Models

Visit: [Google Drive Folder](https://drive.google.com/drive/folders/13r3tEXg5SfyesCLX0w76K5fJ7MHmbJes)

Download:
- `random_forest_synthetic.joblib`
- `random_forest_original.joblib`

Place both files in the project directory.

### Step 2: Run Evaluation Script

```bash
python evaluate_baselines.py
```

This script will:
1. Load the synthetic dataset
2. Prepare the same train/test split used during training
3. Evaluate both models on the test set
4. Compare performance metrics
5. Display comprehensive results

### Step 3: Analyze Results

Key questions to answer:
1. Which model has higher AUC?
2. What's the difference in recall (catching mortality cases)?
3. How do false negative rates compare?
4. Is the performance difference statistically/clinically significant?

## Next Steps After Baseline Understanding

1. **Feature Analysis**: What features are most important in each model?
2. **Error Analysis**: What types of cases does each model miss?
3. **Calibration Analysis**: Are probability predictions well-calibrated?
4. **Improvement Strategy**: How can we beat the baseline?
   - Better feature engineering
   - Different algorithms (XGBoost, Neural Networks)
   - Ensemble methods
   - Better handling of class imbalance
   - Hyperparameter optimization

## Files in This Repository

- `homr.py` - Training script for models
- `evaluate_baselines.py` - Evaluation and comparison script
- `rf_homr_best_hps_synthetic.json` - Hyperparameters for Model 1
- `rf_homr_best_hps_original.json` - Hyperparameters for Model 2
- `csv/dataset.csv` - Synthetic dataset (248,487 visits)
- `csv_handling.py` - Data processing utilities

## References

- Challenge: Synthetic Data for Accessible Learning in Healthcare
- Task: Predicting Hospital One-year Mortality Risk (HOMR)
- Dataset: [Zenodo Record 12954673](https://zenodo.org/records/12954673)
- Pretrained Models: [Google Drive](https://drive.google.com/drive/folders/13r3tEXg5SfyesCLX0w76K5fJ7MHmbJes)

