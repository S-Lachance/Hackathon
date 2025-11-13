# Quick Start Guide: Evaluating Baseline Models

## Overview

This guide helps you evaluate and compare the two baseline models for the Hospital One-year Mortality Risk prediction challenge.

## Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Pretrained Models**
   
   Visit: https://drive.google.com/drive/folders/13r3tEXg5SfyesCLX0w76K5fJ7MHmbJes
   
   Download these files and place them in the project directory:
   - `random_forest_synthetic.joblib` (Model 1: trained on synthetic data)
   - `random_forest_original.joblib` (Model 2: trained on original data)

## Step-by-Step Evaluation

### Step 1: Basic Evaluation

Run the main evaluation script:

```bash
python evaluate_baselines.py
```

**What it does:**
- Loads both pretrained models
- Prepares the test dataset (using the same split as training)
- Evaluates both models with multiple metrics
- Compares performance side-by-side
- Displays hyperparameter differences

**Expected Output:**
- AUC-ROC scores for both models
- Accuracy, Precision, Recall, F1-Score, Specificity
- Confusion matrices
- Performance comparison table
- Overall winner determination

### Step 2: Visualize Results (Optional)

After running the evaluation, you can generate visualizations by integrating the visualization functions into your evaluation script.

**Available Visualizations:**
- ROC curves (comparing discrimination ability)
- Precision-Recall curves (understanding trade-offs)
- Confusion matrices (understanding error types)
- Metric comparison bar charts
- Class weight comparison

## Understanding the Results

### Key Metrics Explained

| Metric | What it Measures | Clinical Relevance |
|--------|-----------------|-------------------|
| **AUC-ROC** | Overall discrimination ability (0.5-1.0) | Higher = better at separating mortality vs survival |
| **Recall (Sensitivity)** | % of mortality cases correctly identified | Critical: missing high-risk patients is costly |
| **Specificity** | % of survival cases correctly identified | Important: too many false alarms waste resources |
| **Precision** | % of mortality predictions that are correct | Affects trust in the system |
| **F1-Score** | Balance between precision and recall | Overall effectiveness |

### Clinical Context

In healthcare prediction:
- **False Negatives (FN)** are typically more costly than False Positives (FP)
  - Missing a high-risk patient can lead to preventable mortality
  - False alarms lead to unnecessary interventions but are less harmful
  
- **Class Imbalance** is normal
  - Most patients don't die within one year (thankfully!)
  - Models must handle this imbalance carefully

### Interpreting Model Comparison

**If Model 1 (Synthetic) performs similarly to Model 2 (Original):**
- ✓ Synthetic data is high-quality and preserves important patterns
- ✓ Safe to use synthetic data for model development
- ✓ Privacy-preserving approach is validated

**If Model 2 (Original) significantly outperforms Model 1:**
- ⚠ Synthetic data may not capture all real-world complexity
- ⚠ May need improved synthetic data generation
- ⚠ Consider what patterns are missing in synthetic data

## Model Details

### Model 1: Trained on Synthetic Dataset

```json
{
    "n_estimators": 1024,
    "min_samples_leaf": 10,
    "max_features": 20,
    "class_weight": {"0": 0.827, "1": 0.173}
}
```

**Characteristics:**
- Large ensemble (1024 trees)
- Conservative class weighting (17.3% on mortality class)
- Trained on privacy-preserving synthetic data

### Model 2: Trained on Original Dataset

```json
{
    "n_estimators": 1024,
    "min_samples_leaf": 10,
    "max_features": 20,
    "class_weight": {"0": 0.742, "1": 0.258}
}
```

**Characteristics:**
- Same tree structure as Model 1
- More aggressive weighting (25.8% on mortality class)
- Trained on real patient data

**Key Difference:** Class weighting strategy
- Model 2 puts ~50% more weight on catching mortality cases
- Suggests different optimal operating point for real vs synthetic data

## What to Look For

### 1. Performance Gap
- How big is the AUC difference?
- Is it statistically/clinically meaningful?
- Where do the models differ most?

### 2. Error Patterns
- Which model has lower false negative rate?
- What's the precision-recall trade-off?
- Are errors systematic or random?

### 3. Calibration
- Do predicted probabilities match actual risks?
- Are both models over/under-confident?

### 4. Feature Importance (Advanced)
- Which features drive predictions?
- Are important features the same for both models?
- Do synthetic patterns match real patterns?

## Next Steps After Evaluation

Once you understand the baseline:

1. **Error Analysis**: Examine misclassified cases
   - What patient profiles are difficult to predict?
   - Are there systematic biases?

2. **Feature Engineering**: Create new features
   - Interaction terms
   - Aggregate statistics
   - Domain-specific transformations

3. **Model Improvements**: Try different approaches
   - XGBoost, LightGBM (gradient boosting)
   - Deep learning (if dataset is large enough)
   - Ensemble methods
   - Better hyperparameter tuning

4. **Better Handling of Imbalance**:
   - SMOTE (synthetic minority oversampling)
   - Cost-sensitive learning
   - Threshold optimization
   - Focal loss

5. **Cross-Validation**: More robust evaluation
   - 5-fold or 10-fold CV
   - Stratified sampling
   - Temporal validation (if time data available)

## Troubleshooting

### Models not found
```
Error: [Errno 2] No such file or directory: 'random_forest_synthetic.joblib'
```
**Solution**: Download the pretrained models from Google Drive (see Prerequisites)

### Dataset not found
```
Error: [Errno 2] No such file or directory: 'csv/dataset.csv'
```
**Solution**: The dataset should be in the `csv/` directory. Download from https://zenodo.org/records/12954673

### Module not found
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

## Files Reference

| File | Purpose |
|------|---------|
| `evaluate_baselines.py` | Main evaluation script |
| `visualize_comparison.py` | Visualization functions |
| `download_models.py` | Helper to download models |
| `BASELINE_ANALYSIS.md` | Detailed analysis documentation |
| `EVALUATION_GUIDE.md` | This file - quick start guide |
| `homr.py` | Original training script |
| `rf_homr_best_hps_*.json` | Optimized hyperparameters |

## Questions to Answer

After running the evaluation, you should be able to answer:

1. ✓ Which model has better overall performance (AUC)?
2. ✓ Which model catches more mortality cases (Recall)?
3. ✓ Which model has fewer false alarms (Specificity)?
4. ✓ How do the class weighting strategies differ?
5. ✓ Is synthetic data adequate for this task?
6. ✓ What's your strategy to improve upon the baseline?

## Support

For issues or questions:
- Review the README.md for general information
- Check BASELINE_ANALYSIS.md for detailed context
- Examine the code comments in evaluate_baselines.py
- Review the challenge documentation

## Success Criteria

You've successfully completed the baseline evaluation when you can:
- ✓ Run both models on the test set
- ✓ Compare their performance across multiple metrics
- ✓ Understand the key differences between them
- ✓ Identify strengths and weaknesses of each
- ✓ Formulate a strategy to improve upon them

Good luck with the challenge! 🚀

