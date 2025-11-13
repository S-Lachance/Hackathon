================================================================================
BASELINE MODEL EVALUATION
RSN Hackathon Challenge: Hospital One-year Mortality Risk Prediction
================================================================================
✓ Pretrained models already exist.

================================================================================
LOADING DATASET
================================================================================
✓ Loaded dataset: 248485 rows, 248 columns
  - Continuous features: 4
  - Binary features: 236
  - Categorical features: 4
✓ Selected one visit per patient: 123646 rows
✓ One-hot encoded categorical features: 37 new columns
✓ Split data: 61823 train, 61823 test
✓ Total features: 277

================================================================================
LOADING PRETRAINED MODELS
================================================================================
C:\Users\sam62\miniconda3\envs\Hackathon\Lib\site-packages\sklearn\base.py:442: InconsistentVersionWarning: Trying to unpickle estimator DecisionTreeClassifier from version 1.6.1 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
C:\Users\sam62\miniconda3\envs\Hackathon\Lib\site-packages\sklearn\base.py:442: InconsistentVersionWarning: Trying to unpickle estimator RandomForestClassifier from version 1.6.1 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
✓ Loaded Model 1: random_forest_synthetic.joblib
✓ Loaded Model 2: random_forest_original.joblib

================================================================================
EVALUATING: Model 1 (Trained on Synthetic Dataset)
================================================================================

Performance Metrics:
  AUC-ROC:     0.8934
  Accuracy:    0.8972
  Precision:   0.9421
  Recall:      0.0371
  F1-Score:    0.0713
  Specificity: 0.9997

Confusion Matrix:
  TN:  55223  FP:     15
  FN:   6341  TP:    244

Class Distribution in Test Set:
  Negative (0):  55238 (89.35%)
  Positive (1):   6585 (10.65%)

================================================================================
EVALUATING: Model 2 (Trained on Original Dataset)
================================================================================

Performance Metrics:
  AUC-ROC:     0.8989
  Accuracy:    0.8967
  Precision:   0.9541
  Recall:      0.0316
  F1-Score:    0.0611
  Specificity: 0.9998

Confusion Matrix:
  TN:  55228  FP:     10
  FN:   6377  TP:    208

Class Distribution in Test Set:
  Negative (0):  55238 (89.35%)
  Positive (1):   6585 (10.65%)

================================================================================
MODEL COMPARISON SUMMARY
================================================================================

Metric          Synthetic       Original        Difference      Better
--------------------------------------------------------------------------------
AUC             0.8934          0.8989                  -0.0055 Original
ACCURACY        0.8972          0.8967                  +0.0005 Synthetic
PRECISION       0.9421          0.9541                  -0.0120 Original
RECALL          0.0371          0.0316                  +0.0055 Synthetic
F1              0.0713          0.0611                  +0.0102 Synthetic
SPECIFICITY     0.9997          0.9998                  -0.0001 Original

================================================================================

OVERALL PERFORMANCE:
  = Tie: Each model wins on 3/6 metrics

================================================================================

================================================================================
HYPERPARAMETERS COMPARISON
================================================================================

Model 1 (Synthetic) Hyperparameters:
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.7424095330133718, '1': 0.2575904669866283}

Model 2 (Original) Hyperparameters:
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.8270747379290138, '1': 0.17292526207098619}

  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.7424095330133718, '1': 0.2575904669866283}

Model 2 (Original) Hyperparameters:
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.7424095330133718, '1': 0.2575904669866283}

Model 2 (Original) Hyperparameters:
  n_estimators: 1024
  min_samples_leaf: 10
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.7424095330133718, '1': 0.2575904669866283}

Model 2 (Original) Hyperparameters:
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.7424095330133718, '1': 0.2575904669866283}

  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.7424095330133718, '1': 0.2575904669866283}

  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.7424095330133718, '1': 0.2575904669866283}
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.7424095330133718, '1': 0.2575904669866283}
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.7424095330133718, '1': 0.2575904669866283}
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  n_estimators: 1024
  min_samples_leaf: 10
  n_estimators: 1024
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.7424095330133718, '1': 0.2575904669866283}

Model 2 (Original) Hyperparameters:
  n_estimators: 1024
  min_samples_leaf: 10
  max_features: 20
  class_weight: {'0': 0.8270747379290138, '1': 0.17292526207098619}

================================================================================
EVALUATION COMPLETE
================================================================================