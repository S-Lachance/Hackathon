import json
import re
from typing import Tuple, List

import joblib
import pandas as pd
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV


def get_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], str]:
    """
    Returns predictors to use for the POYM task.

    Args:
        df (pd.DataFrame): Dataframe containing all variables

    Returns:
        List of predictors to use for the POYM task.
    """

    # Comorbidities diagnostic variables
    DX_COLS = [dx for dx in list(df.columns) if re.compile("^(dx_).+").search(dx)]

    # Admission diagnosis variables
    ADM_COLS = [adm for adm in list(df.columns) if re.compile("^(adm_).+").search(adm)]

    # Demographic, previous care utilization and characteristics of the current admission variables
    CAT_COLS = ['gender', 'living_status', 'admission_group', 'service_group']
    CONT_COLS = ["age_original", "ed_visit_count", "ho_ambulance_count", "total_duration"]
    OTHER_BIN_COLS = ["flu_season", "is_ambulance", "is_icu_start_ho", "is_urg_readm", "has_dx"]

    # Target variable
    OYM = "oym"

    # Binary columns
    BIN_COLS = DX_COLS + ADM_COLS + OTHER_BIN_COLS

    return CONT_COLS, BIN_COLS, CAT_COLS, OYM

# Install the data
# Windows =>
# import urllib.request
# url = "https://zenodo.org/records/12954673/files/dataset.csv"
# output_path = "dataset.csv"
# urllib.request.urlretrieve(url, output_path)

# Linux =>
# subprocess.run(" wget https://zenodo.org/records/12954673/files/dataset.csv", shell=True, check=True)


# Read the data
data = pd.read_csv('dataset.csv')
cont_cols, bin_cols, cat_cols, target = get_cols(data)

# Select one single visit per patient to avoid data leakage
data = data.groupby("patient_id").apply(lambda x: x.sample(1, random_state=42), include_groups=False).reset_index(drop=True)

# One hot encoding for categorical variables
onehot_data = pd.get_dummies(data, columns=cat_cols,  dtype=int)
onehot_cat_cols = [c for c in onehot_data.columns if c not in data.columns]

# Split to learning and holdout set
x_train, x_test, y_train, y_test = train_test_split(onehot_data, data[target], test_size=0.5, random_state=42)


# Define feature columns
feature_cols = cont_cols + bin_cols + onehot_cat_cols

# Define parameter grid for hyperparameter optimization
param_dist = {
    'n_estimators': [128, 256, 512, 1024],
    'max_features': [10, 15, 20],
    'min_samples_leaf': [10, 20, 40, 60, 80],
    'class_weight': [{0: 1 - w, 1: w} for w in uniform.rvs(loc=0.1, scale=0.8, size=100)]
}

# Initialize base model
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Perform random search with cross-validation
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=100,                    # 100 random combinations
    scoring='roc_auc',             # AUC scoring
    cv=5,                          # 5-fold CV
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(x_train[feature_cols], y_train)

# Get best hyperparameters
best_hps = random_search.best_params_

# Save best hyperparameters to JSON
with open("rf_homr_best_hps_synthetic.json", "w") as f:
    json.dump(best_hps, f, indent=4)

# Retrain the model with the best hyperparameters on the full training data
best_model = RandomForestClassifier(**best_hps, random_state=42, n_jobs=-1)
best_model.fit(x_train[feature_cols], y_train)

# Evaluate on the test set
y_pred_proba = best_model.predict_proba(x_test[feature_cols])[:, 1]

# Compute AUC
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Test AUC with best model: {test_auc:.4f}")

# Save the trained model
joblib.dump(best_model, "random_forest_synthetic.joblib")


