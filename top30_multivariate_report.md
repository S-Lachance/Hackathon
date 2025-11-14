# Top 30 Features Multivariate Analysis Report

## Dataset Overview
- Total patients: 123646
- Features analyzed: 23
- Mortality rate: 0.105

## Model Performance
- Logistic Regression AUC: 0.8309
- Risk Score AUC: 0.7635

## Top Risk Factors
| Feature | Odds Ratio | Type |
|---------|------------|------|
| adm_metastasis | 10.59 | binary |
| dx_cancer_ed | 7.06 | binary |
| dx_metastatic_solid_cancer | 4.32 | binary |
| adm_lung_cancer | 3.38 | binary |
| is_urg_readm | 3.13 | binary |
| age_original | 2.58 | continuous |
| adm_cancer | 2.34 | binary |
| dx_chest_cancer_2 | 2.03 | binary |
| is_ambulance | 1.93 | binary |
| dx_obstructive | 1.40 | binary |

## Top Protective Factors
| Feature | Odds Ratio | Type |
|---------|------------|------|
| ho_ambulance_count | 0.99 | continuous |
| dx_cad | 0.99 | binary |
| adm_pregnancy | 0.06 | binary |

## Files Generated
- `top30_descriptive_stats.csv` - Feature statistics
- `top30_logistic_coefficients.csv` - Logistic regression results
- `top30_logistic_coefficients.png` - Coefficient visualization
- `top30_risk_score_analysis.png` - Risk score analysis
