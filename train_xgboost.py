"""
XGBoost Model Training for Hospital Mortality Prediction
Implements improved model with better imbalance handling
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import re
from typing import Tuple, List
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

def get_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], str]:
    """Returns predictors to use for the POYM task."""
    DX_COLS = [dx for dx in list(df.columns) if re.compile("^(dx_).+").search(dx)]
    ADM_COLS = [adm for adm in list(df.columns) if re.compile("^(adm_).+").search(adm)]
    CAT_COLS = ['gender', 'living_status', 'admission_group', 'service_group']
    CONT_COLS = ["age_original", "ed_visit_count", "ho_ambulance_count", "total_duration"]
    OTHER_BIN_COLS = ["flu_season", "is_ambulance", "is_icu_start_ho", "is_urg_readm", "has_dx"]
    BIN_COLS = DX_COLS + ADM_COLS + OTHER_BIN_COLS
    OYM = "oym"
    return CONT_COLS, BIN_COLS, CAT_COLS, OYM

def train_xgboost_model(X_train, y_train, X_val, y_val, hyperparams=None):
    """
    Train XGBoost model with specified hyperparameters
    """
    
    # Calculate class imbalance ratio
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n✓ Class imbalance ratio: {scale_pos_weight:.2f}:1 (negative:positive)")
    print(f"  Using scale_pos_weight={scale_pos_weight:.2f}")
    
    # Default hyperparameters (can be overridden)
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 1000,
            'max_depth': 7,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'auc',
            'early_stopping_rounds': 50,
            'tree_method': 'hist',  # Faster for large datasets
            'random_state': 42
        }
    
    print("\n⏳ Training XGBoost model...")
    print("Hyperparameters:")
    for key, value in hyperparams.items():
        if key != 'early_stopping_rounds':
            print(f"  {key}: {value}")
    
    # Extract early stopping parameter
    early_stopping = hyperparams.pop('early_stopping_rounds', 50)
    
    # Create model
    model = xgb.XGBClassifier(**hyperparams)
    
    # Train with validation monitoring
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score: {model.best_score:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test, model_name="XGBoost"):
    """
    Comprehensive model evaluation
    """
    print("\n" + "="*80)
    print(f"EVALUATING: {model_name}")
    print("="*80)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    # Print results
    print(f"\nPerformance Metrics:")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:>6,}  FP: {fp:>6,}")
    print(f"  FN: {fn:>6,}  TP: {tp:>6,}")
    
    # Class distribution
    n_neg = (y_test == 0).sum()
    n_pos = (y_test == 1).sum()
    print(f"\nClass Distribution in Test Set:")
    print(f"  Negative (0): {n_neg:>6,} ({n_neg/len(y_test)*100:.2f}%)")
    print(f"  Positive (1): {n_pos:>6,} ({n_pos/len(y_test)*100:.2f}%)")
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def plot_feature_importance(model, feature_names, top_n=30):
    """
    Plot feature importance from XGBoost model
    """
    # Get feature importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Plot top features
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]
    
    y_pos = range(len(top_features))
    ax.barh(y_pos, top_importance, color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved feature importance plot: xgboost_feature_importance.png")
    
    # Save to CSV
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importance[indices]
    })
    importance_df.to_csv('xgboost_feature_importance.csv', index=False)
    print("✓ Saved feature importance data: xgboost_feature_importance.csv")

def plot_training_curves(model):
    """
    Plot training and validation AUC curves
    """
    results = model.evals_result()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results['validation_0']['auc'], label='Training AUC', linewidth=2)
    ax.plot(results['validation_1']['auc'], label='Validation AUC', linewidth=2)
    ax.axvline(model.best_iteration, color='red', linestyle='--', 
               label=f'Best Iteration ({model.best_iteration})', linewidth=2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('XGBoost Training Progress', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgboost_training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved training curves: xgboost_training_curves.png")

def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("XGBOOST MODEL TRAINING")
    print("RSN Hackathon Challenge: Hospital One-year Mortality Risk Prediction")
    print("="*80)
    
    # Load dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    data = pd.read_csv('csv/dataset.csv')
    print(f"✓ Loaded dataset: {data.shape[0]:,} rows, {data.shape[1]} columns")
    
    # Get column specifications
    cont_cols, bin_cols, cat_cols, target = get_cols(data)
    print(f"  - Continuous features: {len(cont_cols)}")
    print(f"  - Binary features: {len(bin_cols)}")
    print(f"  - Categorical features: {len(cat_cols)}")
    
    # Select one visit per patient to avoid data leakage
    data = data.groupby("patient_id", group_keys=False).apply(
        lambda x: x.sample(1, random_state=42)
    ).reset_index(drop=True)
    print(f"✓ Selected one visit per patient: {data.shape[0]:,} rows")
    
    # One hot encoding for categorical variables
    onehot_data = pd.get_dummies(data, columns=cat_cols, dtype=int)
    onehot_cat_cols = [c for c in onehot_data.columns if c not in data.columns]
    print(f"✓ One-hot encoded categorical features: {len(onehot_cat_cols)} new columns")
    
    # Prepare features
    feature_cols = cont_cols + bin_cols + onehot_cat_cols
    X = onehot_data[feature_cols].values
    y = onehot_data[target].values
    
    print(f"✓ Total features: {len(feature_cols)}")
    print(f"✓ Target distribution: {y.sum():,} positive ({y.mean()*100:.2f}%)")
    
    # Split data: 50% train+val, 50% test (to match baseline evaluation)
    np.random.seed(42)
    test_indices = np.random.choice(len(X), size=int(0.5*len(X)), replace=False)
    train_val_indices = np.array([i for i in range(len(X)) if i not in test_indices])
    
    X_train_val, X_test = X[train_val_indices], X[test_indices]
    y_train_val, y_test = y[train_val_indices], y[test_indices]
    
    # Further split train_val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print(f"\n✓ Data split:")
    print(f"  Training:   {len(X_train):,} samples ({y_train.sum():,} positive)")
    print(f"  Validation: {len(X_val):,} samples ({y_val.sum():,} positive)")
    print(f"  Test:       {len(X_test):,} samples ({y_test.sum():,} positive)")
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test, "XGBoost Model")
    
    # Plot training curves
    plot_training_curves(model)
    
    # Feature importance
    plot_feature_importance(model, feature_cols, top_n=30)
    
    # Save model
    model_filename = 'xgboost_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\n✓ Model saved: {model_filename}")
    
    # Save metrics
    with open('xgboost_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Metrics saved: xgboost_metrics.json")
    
    # Compare with baseline
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    print("\nBaseline Random Forest (from Evaluation_results.md):")
    print("  AUC-ROC:     0.8989")
    print("  Recall:      0.0316")
    print("  F1-Score:    0.0611")
    print("\nYour XGBoost Model:")
    print(f"  AUC-ROC:     {metrics['auc']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1']:.4f}")
    
    auc_improvement = metrics['auc'] - 0.8989
    recall_improvement = metrics['recall'] - 0.0316
    
    print(f"\nImprovement:")
    print(f"  AUC:         {auc_improvement:+.4f}")
    print(f"  Recall:      {recall_improvement:+.4f} ({recall_improvement/0.0316*100:+.1f}%)")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\n💡 Next Steps:")
    print("  1. Run: python optimize_thresholds.py (update with your XGBoost model)")
    print("  2. Review feature importance to understand what drives predictions")
    print("  3. Iterate: Try different hyperparameters or feature engineering")
    print("\n")

if __name__ == "__main__":
    main()

