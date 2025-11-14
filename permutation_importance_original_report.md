# Permutation Feature Importance Analysis Report (Original Model)

## Overview
This analysis uses permutation feature importance to identify which features are most crucial for predicting mortality (oym=TRUE) and survival (oym=FALSE) using the original model.

**Method**: Permutation importance measures how much model performance (AUC) decreases when each feature is randomly shuffled.

## Dataset
- Total patients: 61,823
- Features analyzed: 277
- Mortality rate: 0.107
- Model: Random Forest (trained on original dataset)

## Top 10 Features by Permutation Importance
| Rank | Feature | Importance | Std Dev | CV Ratio |
|------|---------|------------|--------|----------|
2d2d2d2d2d2d2d2d2d2d
## Interpretation
### What does permutation importance mean?
- **Higher values** = Feature is crucial for accurate predictions
- **Importance score** = Decrease in AUC when feature is randomly permuted
- **Low CV ratio** = Stable, reliable importance scores
- **High CV ratio** = Unstable, variable importance scores

### Comparison with Synthetic Model
- **Age**: Most important feature in both models
- **Cancer features**: Consistently highly ranked
- **Clinical acuity**: Ambulance, urgent admissions prominent
- **Service specialization**: ICU, palliative care, oncology important

## Files Generated
- `permutation_importance_original_mortality_top20.png` - Mortality prediction features
- `permutation_importance_original_survival_top20.png` - Survival prediction features
- `permutation_importance_original_comparison.png` - Side-by-side comparison
- `permutation_importance_original_summary.png` - Summary with confidence intervals
- `original_permutation_importance_detailed.csv` - Complete results
