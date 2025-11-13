# Baseline Models Summary

## TL;DR

You have **two baseline Random Forest models** to evaluate:

| Model | Training Data | Key Characteristic | Class Weight (Mortality) |
|-------|--------------|-------------------|-------------------------|
| **Model 1** | Synthetic dataset | Conservative weighting | 17.3% |
| **Model 2** | Original dataset | Aggressive weighting | 25.8% |

**Your Task**: Evaluate and compare their performance on the test set to understand which approach works better.

---

## Quick Facts

### The Task
- **Predict**: One-year mortality risk after hospital admission
- **Data**: 248,487 patient visits, 246 features
- **Metric**: AUC-ROC (primary), plus Precision, Recall, F1, Specificity
- **Challenge**: Severe class imbalance (most patients survive)

### Model Architecture
Both models use Random Forest with:
- 1,024 decision trees
- Max 20 features per split
- Min 10 samples per leaf
- **Different class weights** (main difference)

### Why Two Models?

**Model 1 (Synthetic)**:
- Shows what's possible with privacy-preserving synthetic data
- Important for healthcare where data access is restricted
- Benchmark for synthetic data quality

**Model 2 (Original)**:
- Trained on real patient data
- Gold standard for comparison
- Represents "best case" with full data access

---

## The Critical Difference: Class Weights

```
Model 1 (Synthetic): Class 0 = 82.7%, Class 1 = 17.3%
Model 2 (Original):  Class 0 = 74.2%, Class 1 = 25.8%
```

**What this means**:
- Model 2 puts 50% more emphasis on catching mortality cases
- Suggests different optimal strategy for real vs synthetic data
- Likely reflects different class distributions or costs

---

## What You Need to Do

### 1. Download Models ⬇️
Get from: https://drive.google.com/drive/folders/13r3tEXg5SfyesCLX0w76K5fJ7MHmbJes
- `random_forest_synthetic.joblib`
- `random_forest_original.joblib`

### 2. Run Evaluation 🏃
```bash
pip install -r requirements.txt
python evaluate_baselines.py
```

### 3. Analyze Results 📊
Answer these questions:
- Which model has higher AUC?
- Which catches more mortality cases (Recall)?
- Which has fewer false alarms (Specificity)?
- Is the performance gap significant?
- Is synthetic data adequate?

---

## Expected Insights

### Scenario A: Models Perform Similarly
**Implication**: Synthetic data is high-quality
- ✓ Can safely use synthetic data for development
- ✓ Privacy-preserving approach validated
- ✓ Focus on improving model architecture

### Scenario B: Original Model Significantly Better
**Implication**: Synthetic data missing key patterns
- ⚠ Need better synthetic data generation
- ⚠ May need to identify what's missing
- ⚠ Consider hybrid approaches

### Scenario C: Synthetic Model Better
**Implication**: Unexpected but interesting
- 🤔 Original model might be overfitting
- 🤔 Synthetic data might remove noise
- 🤔 Class weighting might be suboptimal for original data

---

## Performance Metrics Guide

| Metric | Best for Understanding | Clinical Importance |
|--------|----------------------|-------------------|
| **AUC-ROC** | Overall model quality | ⭐⭐⭐⭐⭐ Primary metric |
| **Recall** | How many deaths we catch | ⭐⭐⭐⭐⭐ Critical - missed deaths costly |
| **Specificity** | How many survivors correctly ID'd | ⭐⭐⭐ Important - false alarms costly |
| **Precision** | Prediction accuracy | ⭐⭐⭐ Trust in system |
| **F1-Score** | Balance of precision/recall | ⭐⭐⭐ Overall effectiveness |

---

## Key Files

| File | What It Does |
|------|-------------|
| `evaluate_baselines.py` | 🎯 Run this to evaluate models |
| `EVALUATION_GUIDE.md` | 📖 Detailed step-by-step guide |
| `BASELINE_ANALYSIS.md` | 📊 In-depth analysis and context |
| `visualize_comparison.py` | 📈 Visualization functions |
| `rf_homr_best_hps_*.json` | ⚙️ Model hyperparameters |

---

## After Baseline Evaluation

Once you understand the baseline, you can:

1. **Beat the Baseline**
   - Try XGBoost, LightGBM, or neural networks
   - Better feature engineering
   - Improved hyperparameter tuning
   - Ensemble methods

2. **Deep Dive Analysis**
   - Feature importance analysis
   - Error pattern investigation
   - Calibration assessment
   - Subgroup analysis

3. **Production Considerations**
   - Threshold optimization for clinical use
   - Model interpretability
   - Deployment strategy
   - Monitoring plan

---

## Success Checklist

- [ ] Downloaded both pretrained models
- [ ] Installed all dependencies
- [ ] Ran evaluation script successfully
- [ ] Compared AUC scores
- [ ] Analyzed confusion matrices
- [ ] Understood recall vs specificity trade-offs
- [ ] Identified key differences between models
- [ ] Formulated improvement strategy

---

## Remember

**In healthcare ML**:
- False negatives (missing high-risk patients) are typically more costly than false positives
- Class imbalance is normal and must be handled carefully
- Model interpretability matters for clinical adoption
- Synthetic data quality is crucial for privacy-preserving ML

**For this challenge**:
- The goal is to understand baseline performance FIRST
- Then identify opportunities for improvement
- Finally, build a better model
- Document your insights along the way

---

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Evaluate baselines (after downloading models)
python evaluate_baselines.py

# Check if models exist
python download_models.py
```

---

## Key Insights from Hyperparameters

### Similarities (Same Architecture)
- Both use 1024 trees (large, stable ensemble)
- Both use max 20 features (controlled complexity)
- Both use min 10 samples/leaf (prevents overfitting)

### Differences (Strategy)
- **Class weights differ by ~50%** ← This is the main story
- Reflects different optimal operating points
- Shows how data source affects model calibration

---

## Bottom Line

You're comparing two philosophically different approaches:
1. **Model 1**: "Can we do good ML with privacy-safe synthetic data?"
2. **Model 2**: "How good can we get with real data?"

The answer tells you:
- Whether synthetic data is "good enough"
- Where improvements are needed
- What your next steps should be

Now go evaluate those models! 🚀

