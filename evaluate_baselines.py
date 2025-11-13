"""
Evaluation script for comparing the two baseline models:
- Model 1: Trained on synthetic dataset
- Model 2: Trained on original dataset
"""

import json
import os
import re
from typing import Tuple, List

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split


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


def download_pretrained_models():
    """Download pretrained models from Google Drive if they don't exist."""
    
    # Check if models already exist
    if os.path.exists("random_forest_synthetic.joblib") and os.path.exists("random_forest_original.joblib"):
        print("✓ Pretrained models already exist.")
        return True
    
    print("\n" + "="*80)
    print("DOWNLOADING PRETRAINED MODELS")
    print("="*80)
    print("\nThe pretrained models need to be downloaded from Google Drive:")
    print("https://drive.google.com/drive/u/4/folders/13r3tEXg5SfyesCLX0w76K5fJ7MHmbJes")
    print("\nPlease download the following files and place them in this directory:")
    print("  - random_forest_synthetic.joblib")
    print("  - random_forest_original.joblib")
    print("\nAlternatively, you can use gdown to download them programmatically.")
    print("="*80)
    
    return False


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate a model on test data and return comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print(f"\nPerformance Metrics:")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:>6}  FP: {fp:>6}")
    print(f"  FN: {fn:>6}  TP: {tp:>6}")
    
    print(f"\nClass Distribution in Test Set:")
    print(f"  Negative (0): {(y_test == 0).sum():>6} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")
    print(f"  Positive (1): {(y_test == 1).sum():>6} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")
    
    return {
        'model_name': model_name,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def compare_models(results_synthetic, results_original):
    """Print comparison between the two models."""
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1', 'specificity']
    
    print(f"\n{'Metric':<15} {'Synthetic':<15} {'Original':<15} {'Difference':<15} {'Better':<10}")
    print("-" * 80)
    
    for metric in metrics:
        syn_val = results_synthetic[metric]
        orig_val = results_original[metric]
        diff = syn_val - orig_val
        better = "Synthetic" if diff > 0 else ("Original" if diff < 0 else "Tie")
        
        print(f"{metric.upper():<15} {syn_val:<15.4f} {orig_val:<15.4f} {diff:+15.4f} {better:<10}")
    
    print("\n" + "="*80)
    
    # Determine overall winner
    syn_wins = sum(1 for m in metrics if results_synthetic[m] > results_original[m])
    orig_wins = sum(1 for m in metrics if results_original[m] > results_synthetic[m])
    
    print("\nOVERALL PERFORMANCE:")
    if syn_wins > orig_wins:
        print(f"  ✓ Model 1 (Synthetic) wins on {syn_wins}/{len(metrics)} metrics")
    elif orig_wins > syn_wins:
        print(f"  ✓ Model 2 (Original) wins on {orig_wins}/{len(metrics)} metrics")
    else:
        print(f"  = Tie: Each model wins on {syn_wins}/{len(metrics)} metrics")
    
    print("\n" + "="*80)


def main():
    """Main evaluation function."""
    
    print("\n" + "="*80)
    print("BASELINE MODEL EVALUATION")
    print("RSN Hackathon Challenge: Hospital One-year Mortality Risk Prediction")
    print("="*80)
    
    # Step 1: Check for pretrained models
    if not download_pretrained_models():
        print("\n⚠ Please download the pretrained models first and run this script again.")
        return
    
    # Step 2: Load the dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    data = pd.read_csv('csv/dataset.csv')
    print(f"✓ Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Get column specifications
    cont_cols, bin_cols, cat_cols, target = get_cols(data)
    print(f"  - Continuous features: {len(cont_cols)}")
    print(f"  - Binary features: {len(bin_cols)}")
    print(f"  - Categorical features: {len(cat_cols)}")
    
    # Select one visit per patient to avoid data leakage
    data = data.groupby("patient_id").apply(
        lambda x: x.sample(1, random_state=42), 
        include_groups=False
    ).reset_index(drop=True)
    print(f"✓ Selected one visit per patient: {data.shape[0]} rows")
    
    # One hot encoding for categorical variables
    onehot_data = pd.get_dummies(data, columns=cat_cols, dtype=int)
    onehot_cat_cols = [c for c in onehot_data.columns if c not in data.columns]
    print(f"✓ One-hot encoded categorical features: {len(onehot_cat_cols)} new columns")
    
    # Split to learning and holdout set (same as training)
    x_train, x_test, y_train, y_test = train_test_split(
        onehot_data, 
        data[target], 
        test_size=0.5, 
        random_state=42
    )
    print(f"✓ Split data: {x_train.shape[0]} train, {x_test.shape[0]} test")
    
    # Define feature columns
    feature_cols = cont_cols + bin_cols + onehot_cat_cols
    print(f"✓ Total features: {len(feature_cols)}")
    
    # Step 3: Load pretrained models
    print("\n" + "="*80)
    print("LOADING PRETRAINED MODELS")
    print("="*80)
    
    try:
        rf_synthetic = joblib.load("random_forest_synthetic.joblib")
        print("✓ Loaded Model 1: random_forest_synthetic.joblib")
        
        rf_original = joblib.load("random_forest_original.joblib")
        print("✓ Loaded Model 2: random_forest_original.joblib")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure both model files are in the current directory:")
        print("  - random_forest_synthetic.joblib")
        print("  - random_forest_original.joblib")
        return
    
    # Step 4: Evaluate Model 1 (Synthetic)
    results_synthetic = evaluate_model(
        rf_synthetic, 
        x_test[feature_cols], 
        y_test, 
        "Model 1 (Trained on Synthetic Dataset)"
    )
    
    # Step 5: Evaluate Model 2 (Original)
    results_original = evaluate_model(
        rf_original, 
        x_test[feature_cols], 
        y_test, 
        "Model 2 (Trained on Original Dataset)"
    )
    
    # Step 6: Compare models
    compare_models(results_synthetic, results_original)
    
    # Step 7: Load and display hyperparameters
    print("\n" + "="*80)
    print("HYPERPARAMETERS COMPARISON")
    print("="*80)
    
    with open("rf_homr_best_hps_synthetic.json", "r") as f:
        hps_synthetic = json.load(f)
    
    with open("rf_homr_best_hps_original.json", "r") as f:
        hps_original = json.load(f)
    
    print("\nModel 1 (Synthetic) Hyperparameters:")
    for key, value in hps_synthetic.items():
        print(f"  {key}: {value}")
    
    print("\nModel 2 (Original) Hyperparameters:")
    for key, value in hps_original.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

