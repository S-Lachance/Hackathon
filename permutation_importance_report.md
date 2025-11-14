# Permutation Feature Importance Analysis Report

## Overview
This analysis uses permutation feature importance to identify which features are most crucial for predicting mortality (oym=TRUE) and survival (oym=FALSE).

**Method**: Permutation importance measures how much model performance (AUC) decreases when each feature is randomly shuffled.

## Dataset
- Total patients: 61823
- Features analyzed: 277
- Mortality rate: 0.107
- Model: Random Forest (trained on synthetic dataset)

## Top 10 Features for Mortality Prediction (oym=TRUE)
| Rank | Feature | Importance | Std Dev |
|------|---------|------------|--------|
.4f.4f.4f.4f.4f.4f.4f.4f.4f.4f
## Top 10 Features for Survival Prediction (oym=FALSE)
| Rank | Feature | Importance | Std Dev |
|------|---------|------------|--------|
.4f.4f.4f.4f.4f.4f.4f.4f.4f.4f
## Interpretation
### What does permutation importance mean?
- **Higher positive values** = Feature is crucial for accurate predictions
- **Importance score** = Decrease in AUC when feature is randomly permuted
- **Features with high importance for mortality** help identify high-risk patients
- **Features with high importance for survival** help identify low-risk patients

### Clinical Implications
- Focus clinical attention on patients with high scores on mortality-associated features
- Use survival-associated features to identify patients who may not need intensive monitoring
- These features capture complex interactions that simple correlation might miss

## Files Generated
- `permutation_importance_mortality.csv` - Full mortality importance results
- `permutation_importance_survival.csv` - Full survival importance results
- `permutation_importance_mortality_top20.png` - Mortality importance visualization
- `permutation_importance_survival_top20.png` - Survival importance visualization
- `permutation_importance_comparison.png` - Side-by-side comparison
