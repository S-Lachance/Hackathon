"""
Simple demo script to run predictions on the dataset using the trained models.
Shows how to load the model, prepare data, and make predictions.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from csv_handling import load_and_prep_data
import re

def get_cols(df: pd.DataFrame):
    """Returns predictors to use for the POYM task."""
    DX_COLS = [dx for dx in list(df.columns) if re.compile("^(dx_).+").search(dx)]
    ADM_COLS = [adm for adm in list(df.columns) if re.compile("^(adm_).+").search(adm)]
    CAT_COLS = ['gender', 'living_status', 'admission_group', 'service_group']
    CONT_COLS = ["age_original", "ed_visit_count", "ho_ambulance_count", "total_duration"]
    OTHER_BIN_COLS = ["flu_season", "is_ambulance", "is_icu_start_ho", "is_urg_readm", "has_dx"]
    BIN_COLS = DX_COLS + ADM_COLS + OTHER_BIN_COLS
    OYM = "oym"
    return CONT_COLS, BIN_COLS, CAT_COLS, OYM

def prepare_data_for_prediction(data_path):
    """Load and prepare data for model prediction."""
    print("Loading and preparing data...")

    # Load data
    df = load_and_prep_data(data_path)
    if df is None:
        return None, None

    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Get column specifications
    cont_cols, bin_cols, cat_cols, target = get_cols(df)

    # Select one visit per patient (same as training)
    df = df.groupby("patient_id", group_keys=False).apply(
        lambda x: x.sample(1, random_state=42)
    ).reset_index(drop=True)
    print(f"Selected one visit per patient: {len(df)} rows")

    # One-hot encode categorical variables
    onehot_data = pd.get_dummies(df, columns=cat_cols, dtype=int)
    onehot_cat_cols = [c for c in onehot_data.columns if c not in df.columns]
    print(f"One-hot encoded categorical features: {len(onehot_cat_cols)} new columns")

    # Define feature columns (same as training)
    feature_cols = cont_cols + bin_cols + onehot_cat_cols
    print(f"Total features: {len(feature_cols)}")

    # Prepare features and target
    X = onehot_data[feature_cols]
    y = df[target] if target in df.columns else None

    return X, y

def run_predictions(model_path, X, optimal_threshold=0.076):
    """Run predictions using the specified model."""
    print(f"\nLoading model from {model_path}...")
    model = joblib.load(model_path)

    print("Making predictions...")
    # Get probability predictions
    y_proba = model.predict_proba(X)[:, 1]

    # Apply optimal threshold for clinical decision making
    y_pred = (y_proba >= optimal_threshold).astype(int)

    print(".3f")
    print(f"Patients flagged as high-risk: {y_pred.sum()} out of {len(y_pred)} ({y_pred.sum()/len(y_pred)*100:.1f}%)")

    return y_proba, y_pred

def main():
    """Main demo function."""
    print("=" * 80)
    print("MODEL PREDICTION DEMO")
    print("RSN Hackathon Challenge: Hospital One-year Mortality Risk Prediction")
    print("=" * 80)

    # Load and prepare data
    X, y = prepare_data_for_prediction('csv/dataset.csv')
    if X is None:
        print("Failed to load data")
        return

    # Choose model (synthetic performs slightly better)
    model_path = 'random_forest_synthetic.joblib'

    # Run predictions with optimized threshold for 1:1 alert ratio
    y_proba, y_pred = run_predictions(model_path, X, optimal_threshold=0.076)

    # Show some example predictions
    print("\nExample Predictions:")
    print("-" * 50)
    sample_indices = np.random.choice(len(X), 5, replace=False)

    for i, idx in enumerate(sample_indices):
        prob = y_proba[idx]
        pred = y_pred[idx]
        risk_level = "HIGH RISK" if pred == 1 else "Low Risk"
        print("2d"
              "5.3f"
              "12s")

    # If we have ground truth labels, show performance
    if y is not None:
        print("\nPerformance on Full Dataset:")
        print("-" * 40)
        auc = roc_auc_score(y, y_proba)
        print(".4f")
        print(f"Optimal threshold used: 0.076 (1:1 alert ratio)")
        print(f"Patients flagged: {y_pred.sum()} ({y_pred.sum()/len(y_pred)*100:.1f}%)")

        # Show confusion matrix
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\n💡 Key Insights:")
    print("  • The model outputs probabilities (0.0-1.0)")
    print("  • Threshold = 0.076 gives 1:1 false alerts to true alerts")
    print("  • This catches ~57% of high-risk patients while maintaining clinical utility")
    print("  • Check threshold_optimization_*.png for full trade-off analysis")

if __name__ == "__main__":
    main()
