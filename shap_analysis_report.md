# SHAP Analysis Report: Feature Contributions to Mortality

## Overview
SHAP (SHapley Additive exPlanations) analysis quantifies how much each feature contributes to the model's prediction for each individual instance.

**Method**: TreeExplainer on Random Forest model
**Sample size**: 1000 instances analyzed

## Top 20 Features by Mean Absolute SHAP Value
Features with largest overall impact on predictions:

| Rank | Feature | Mean |SHAP| | Direction | Std Dev |
|------|---------|--------|----------|--------|
2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d
## Top 10 Features Contributing to Mortality (oym=TRUE)
Features that most increase predicted mortality risk:

| Rank | Feature | Mean SHAP |
|------|---------|-----------|
2d2d2d2d2d2d2d2d2d2d
## Top 10 Features Contributing to Survival (oym=FALSE)
Features that most decrease predicted mortality risk:

| Rank | Feature | Mean SHAP |
|------|---------|-----------|
2d2d2d2d2d2d2d2d2d2d
## Interpretation
### SHAP Values Meaning:
- **Positive SHAP**: Feature pushes prediction toward mortality (oym=TRUE)
- **Negative SHAP**: Feature pushes prediction toward survival (oym=FALSE)
- **Magnitude**: Strength of the feature's contribution

### Clinical Applications:
- **Risk factors**: Features with consistently positive SHAP values
- **Protective factors**: Features with consistently negative SHAP values
- **Individual predictions**: SHAP explains why each patient receives their risk score

## Files Generated
- `shap_mortality_features.csv` - Full SHAP analysis for mortality
- `shap_survival_features.csv` - Full SHAP analysis for survival
- `shap_summary_plot.png` - SHAP summary visualization
- `shap_bar_plot.png` - Feature importance bar chart
- `shap_waterfall_high_risk.png` - Example high-risk prediction explanation
- `shap_waterfall_low_risk.png` - Example low-risk prediction explanation
