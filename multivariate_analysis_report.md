# Multivariate Analysis Report: Hospital Mortality Prediction

## Dataset Overview
- Total patients: 123646
- Mortality rate: 0.105 (12995 deaths)

## Model Performance
- Logistic Regression AUC: 0.8573

## Top Correlated Features
| Feature | Correlation | Method |
|---------|-------------|--------|
| age_original | 0.271 | pearson |
| admission_group_urgent | 0.202 | point-biserial |
| is_urg_readm | 0.190 | point-biserial |
| is_ambulance | 0.188 | point-biserial |
| adm_metastasis | 0.180 | point-biserial |
| service_group_palliative_care | 0.159 | point-biserial |
| ed_visit_count | 0.157 | pearson |
| dx_cancer_ed | 0.155 | point-biserial |
| adm_lung_cancer | 0.149 | point-biserial |
| dx_metastatic_solid_cancer | 0.148 | point-biserial |

## Key Predictors from Logistic Regression
### Risk Factors (Odds Ratio > 1)
| Feature | Odds Ratio |
|---------|------------|
| service_group_palliative_care | 10.68 |
| adm_metastasis | 8.35 |
| service_group_icu | 5.39 |
| dx_cancer_ed | 4.17 |
| dx_metastatic_solid_cancer | 3.33 |
| service_group_respirology | 3.08 |
| service_group_hematology_oncology | 2.88 |
| age_original | 2.65 |
| is_urg_readm | 2.51 |
| adm_cancer | 2.46 |

### Protective Factors (Odds Ratio < 1)
| Feature | Odds Ratio |
|---------|------------|
| service_group_family_medicine | 0.92 |
| admission_group_obstetrics | 0.52 |
| adm_pregnancy | 0.49 |
| service_group_obstetrics | 0.44 |

## Analysis Files Generated
- `multivariate_univariate_results.csv` - Univariate statistical tests
- `multivariate_correlations.csv` - Feature correlations with mortality
- `multivariate_logistic_coefficients.csv` - Logistic regression coefficients
- `multivariate_feature_importance.csv` - Multi-method feature importance
- `multivariate_pca_analysis.png` - PCA dimensionality reduction plots
