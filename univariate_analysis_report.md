# Univariate Logistic Regression Analysis Report

## Overview
This analysis performs univariate logistic regression on all features to assess their individual association with mortality (oym=TRUE).

**Methodology**: Each feature is tested individually against the outcome using logistic regression.

## Dataset
- Total features analyzed: 276
- Features with p < 0.1: 224
- Models converged: 270/276

## Top 10 Risk Factors
| Rank | Feature | Odds Ratio | p-value | 95% CI |
|------|---------|------------|---------|--------|
2d2d2d2d2d2d2d2d2d2d
## Top 10 Protective Factors
| Rank | Feature | Odds Ratio | p-value | 95% CI |
|------|---------|------------|---------|--------|
2d2d2d2d2d2d2d2d2d2d
## Statistical Summary
- Mean odds ratio: 4.183
- Median p-value: 5.64e-12
- Features with OR > 2: 148
- Features with OR < 0.5: 53

## Interpretation
### Odds Ratios
- **OR > 1**: Risk factor (increases mortality probability)
- **OR < 1**: Protective factor (decreases mortality probability)
- **OR = 1**: No association

### p-values
- **p < 0.05**: Statistically significant
- **p < 0.01**: Highly significant
- **p < 0.001**: Very highly significant

### Feature Selection
Features with p < 0.1 were selected for multivariate analysis to balance statistical significance with the risk of missing potentially important variables.

## Files Generated
- `univariate_logistic_results.csv` - Complete univariate results
- `univariate_pvalue_distribution.png` - p-value distribution plots
- `univariate_odds_ratios_analysis.png` - Odds ratios and volcano plot
- `univariate_significance_ranking.png` - Statistical significance ranking
