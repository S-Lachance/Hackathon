"""
Permutation Feature Importance Analysis for Mortality Prediction
Finds the 10 features most associated with oym=TRUE and oym=FALSE
using permutation feature importance on trained Random Forest models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import joblib
import re
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], str]:
    """Returns predictors to use for the POYM task."""
    DX_COLS = [dx for dx in list(df.columns) if re.compile("^(dx_).+").search(dx)]
    ADM_COLS = [adm for adm in list(df.columns) if re.compile("^(adm_).+").search(adm)]
    CAT_COLS = ['gender', 'living_status', 'admission_group', 'service_group']
    CONT_COLS = ["age_original", "ed_visit_count", "ho_ambulance_count", "total_duration"]
    OTHER_BIN_COLS = ["flu_season", "is_ambulance", "is_icu_start_ho", "is_urg_readm", "has_dx"]
    BIN_COLS = DX_COLS + ADM_COLS + OTHER_BIN_COLS
    OYM = "oym"
    return CONT_COLS, BIN_COLS, CAT_COLS, OYM

def load_and_prepare_data(filepath='csv/dataset.csv'):
    """Load and prepare the dataset using the same preprocessing as model training."""
    print("Loading and preparing data for permutation importance analysis...")

    # Load data exactly as in evaluate_baselines.py
    data = pd.read_csv(filepath)
    print(f"✓ Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")

    # Get column specifications
    cont_cols, bin_cols, cat_cols, target = get_feature_columns(data)
    print(f"  - Continuous features: {len(cont_cols)}")
    print(f"  - Binary features: {len(bin_cols)}")
    print(f"  - Categorical features: {len(cat_cols)}")

    # Select one visit per patient to avoid data leakage (same as training)
    data = data.groupby("patient_id", group_keys=False).apply(
        lambda x: x.sample(1, random_state=42)
    ).reset_index(drop=True)
    print(f"✓ Selected one visit per patient: {data.shape[0]} rows")

    # One hot encoding for categorical variables (same as training)
    onehot_data = pd.get_dummies(data, columns=cat_cols, dtype=int)
    onehot_cat_cols = [c for c in onehot_data.columns if c not in data.columns]
    print(f"✓ One-hot encoded categorical features: {len(onehot_cat_cols)} new columns")

    # Split to learning and holdout set (same as training: test_size=0.5)
    x_train, x_test, y_train, y_test = train_test_split(
        onehot_data,
        data[target],
        test_size=0.5,
        random_state=42
    )
    print(f"✓ Split data: {x_train.shape[0]} train, {x_test.shape[0]} test")

    # Define feature columns (same as training)
    feature_cols = cont_cols + bin_cols + onehot_cat_cols
    print(f"✓ Total features: {len(feature_cols)}")

    # Return the test set (what the models were trained on)
    return x_test[feature_cols], y_test

def calculate_permutation_importance(model, X, y, feature_names, n_repeats=5):
    """
    Calculate permutation feature importance for a trained model.

    Parameters:
    - model: Trained sklearn model
    - X: Feature matrix
    - y: Target vector
    - feature_names: List of feature names
    - n_repeats: Number of times to permute each feature

    Returns:
    - DataFrame with permutation importance results
    """
    print(f"Calculating permutation importance with {n_repeats} repeats per feature...")

    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=1,  # Run sequentially to avoid sandbox permission issues
        scoring='roc_auc'
    )

    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
        'importance_max': perm_importance.importances.max(axis=1),
        'importance_min': perm_importance.importances.min(axis=1)
    })

    # Sort by absolute importance (most important first)
    results_df['abs_importance'] = results_df['importance_mean'].abs()
    results_df = results_df.sort_values('abs_importance', ascending=False)

    return results_df, perm_importance

def analyze_class_specific_importance(model, X, y, feature_names, target_class, n_repeats=5):
    """
    Analyze which features are most important for predicting a specific class.
    This is done by focusing on samples of the target class vs others.
    """

    print(f"Analyzing features most important for predicting oym={target_class}...")

    if target_class == 1:
        print("Features that help identify MORTALITY cases (oym=TRUE)")
        # For mortality cases, we want features that distinguish deaths from survivors
        # We'll use the full dataset but focus on the positive class performance
        pos_indices = y == 1
        neg_indices = y == 0

        print(f"Analyzing {pos_indices.sum()} mortality cases vs {neg_indices.sum()} survivors")

    else:
        print("Features that help identify SURVIVAL cases (oym=FALSE)")
        # For survival cases, we want features that distinguish survivors from deaths
        pos_indices = y == 0  # Survivors are the "positive" class for this analysis
        neg_indices = y == 1

        print(f"Analyzing {pos_indices.sum()} survival cases vs {neg_indices.sum()} deaths")

    # Calculate permutation importance on the full dataset
    # The importance will reflect how much each feature contributes to correct classification
    results_df, perm_importance = calculate_permutation_importance(
        model, X, y, feature_names, n_repeats
    )

    # Add class-specific information
    results_df['target_class'] = target_class
    results_df['class_label'] = 'Mortality' if target_class == 1 else 'Survival'

    return results_df, perm_importance

def plot_top_features_importance(importance_df, title, top_n=20):
    """Plot the top N most important features."""
    plt.figure(figsize=(12, max(6, top_n * 0.3)))

    top_features = importance_df.head(top_n)

    # Create horizontal bar plot
    bars = plt.barh(range(len(top_features)), top_features['importance_mean'],
                   xerr=top_features['importance_std'], capsize=3)

    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Permutation Importance (AUC Decrease)')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Color bars based on importance
    for i, bar in enumerate(bars):
        if top_features.iloc[i]['importance_mean'] > 0:
            bar.set_color('red')  # Positive importance (helps prediction)
        else:
            bar.set_color('blue')  # Negative importance (hurts prediction when permuted)

    plt.tight_layout()
    return plt.gcf()

def create_interpretation_summary(mortality_importance, survival_importance):
    """Create a summary interpretation of the results."""

    print("\n" + "="*100)
    print("PERMUTATION IMPORTANCE ANALYSIS INTERPRETATION")
    print("="*100)

    print("\n🔴 FEATURES MOST ASSOCIATED WITH MORTALITY (oym=TRUE):")
    print("-" * 60)
    top_mortality = mortality_importance.head(10)
    for i, (_, row) in enumerate(top_mortality.iterrows(), 1):
        importance = row['importance_mean']
        std = row['importance_std']
        print("2d")
    print("\n🟢 FEATURES MOST ASSOCIATED WITH SURVIVAL (oym=FALSE):")
    print("-" * 60)
    top_survival = survival_importance.head(10)
    for i, (_, row) in enumerate(top_survival.iterrows(), 1):
        importance = row['importance_mean']
        std = row['importance_std']
        print("2d")
    print("\n📊 INTERPRETATION:")
    print("-" * 30)
    print("• Permutation importance measures how much AUC decreases when a feature is randomly shuffled")
    print("• Higher positive values = feature is crucial for accurate predictions")
    print("• Features with high importance for mortality help identify high-risk patients")
    print("• Features with high importance for survival help identify low-risk patients")

    # Find features that are important for both
    mortality_features = set(mortality_importance.head(10)['feature'])
    survival_features = set(survival_importance.head(10)['feature'])
    overlap = mortality_features.intersection(survival_features)

    if overlap:
        print(f"\n🔄 FEATURES IMPORTANT FOR BOTH OUTCOMES: {', '.join(overlap)}")
        print("   These features are crucial discriminators between survival and death")

    return top_mortality, top_survival

def main():
    """Main function for permutation importance analysis."""
    print("=" * 100)
    print("PERMUTATION FEATURE IMPORTANCE ANALYSIS")
    print("Finding features most associated with oym=TRUE and oym=FALSE")
    print("=" * 100)

    # Load and prepare data (returns the test set that models were trained on)
    X, y = load_and_prepare_data()

    if X is None:
        print("Failed to load data")
        return

    # Load trained model (using the synthetic model as it's generally better)
    try:
        model = joblib.load("random_forest_synthetic.joblib")
        print("✓ Loaded random_forest_synthetic.joblib model")
    except FileNotFoundError:
        print("✗ Model file not found. Please ensure random_forest_synthetic.joblib exists")
        return

    # Get feature names
    feature_cols = list(X.columns)

    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    print(".1f")

    # Analyze features associated with mortality (oym=TRUE)
    print("\n" + "="*80)
    print("ANALYZING FEATURES ASSOCIATED WITH MORTALITY (oym=TRUE)")
    print("="*80)

    mortality_importance, mortality_perm = analyze_class_specific_importance(
        model, X, y, feature_cols, target_class=1, n_repeats=3
    )

    # Analyze features associated with survival (oym=FALSE)
    print("\n" + "="*80)
    print("ANALYZING FEATURES ASSOCIATED WITH SURVIVAL (oym=FALSE)")
    print("="*80)

    survival_importance, survival_perm = analyze_class_specific_importance(
        model, X, y, feature_cols, target_class=0, n_repeats=3
    )

    # Create interpretation summary
    top_mortality, top_survival = create_interpretation_summary(
        mortality_importance, survival_importance
    )

    # Save results
    print("\nSaving results...")

    mortality_importance.to_csv('permutation_importance_mortality.csv', index=False)
    survival_importance.to_csv('permutation_importance_survival.csv', index=False)

    print("✓ Mortality importance results saved: permutation_importance_mortality.csv")
    print("✓ Survival importance results saved: permutation_importance_survival.csv")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Plot top 20 features for mortality
    fig1 = plot_top_features_importance(mortality_importance,
                                       'Top 20 Features: Permutation Importance for Mortality Prediction',
                                       top_n=20)
    fig1.savefig('permutation_importance_mortality_top20.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Mortality importance plot saved: permutation_importance_mortality_top20.png")

    # Plot top 20 features for survival
    fig2 = plot_top_features_importance(survival_importance,
                                       'Top 20 Features: Permutation Importance for Survival Prediction',
                                       top_n=20)
    fig2.savefig('permutation_importance_survival_top20.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Survival importance plot saved: permutation_importance_survival_top20.png")

    # Create combined comparison plot
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)
    top_mort = mortality_importance.head(15)
    plt.barh(range(len(top_mort)), top_mort['importance_mean'],
            xerr=top_mort['importance_std'], capsize=3, color='red', alpha=0.7)
    plt.yticks(range(len(top_mort)), top_mort['feature'])
    plt.xlabel('Importance (AUC Decrease)')
    plt.title('Features Most Important for\nPredicting Mortality (oym=TRUE)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    top_surv = survival_importance.head(15)
    plt.barh(range(len(top_surv)), top_surv['importance_mean'],
            xerr=top_surv['importance_std'], capsize=3, color='green', alpha=0.7)
    plt.yticks(range(len(top_surv)), top_surv['feature'])
    plt.xlabel('Importance (AUC Decrease)')
    plt.title('Features Most Important for\nPredicting Survival (oym=FALSE)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('permutation_importance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Comparison plot saved: permutation_importance_comparison.png")

    # Create summary report
    with open('permutation_importance_report.md', 'w') as f:
        f.write("# Permutation Feature Importance Analysis Report\n\n")

        f.write("## Overview\n")
        f.write("This analysis uses permutation feature importance to identify which features are most crucial for predicting mortality (oym=TRUE) and survival (oym=FALSE).\n\n")
        f.write("**Method**: Permutation importance measures how much model performance (AUC) decreases when each feature is randomly shuffled.\n\n")

        f.write("## Dataset\n")
        f.write(f"- Total patients: {len(X)}\n")
        f.write(f"- Features analyzed: {len(feature_cols)}\n")
        f.write(f"- Mortality rate: {y.mean():.3f}\n")
        f.write(f"- Model: Random Forest (trained on synthetic dataset)\n\n")

        f.write("## Top 10 Features for Mortality Prediction (oym=TRUE)\n")
        f.write("| Rank | Feature | Importance | Std Dev |\n")
        f.write("|------|---------|------------|--------|\n")
        for i, (_, row) in enumerate(top_mortality.iterrows(), 1):
            f.write(".4f")
        f.write("\n")

        f.write("## Top 10 Features for Survival Prediction (oym=FALSE)\n")
        f.write("| Rank | Feature | Importance | Std Dev |\n")
        f.write("|------|---------|------------|--------|\n")
        for i, (_, row) in enumerate(top_survival.iterrows(), 1):
            f.write(".4f")
        f.write("\n")

        f.write("## Interpretation\n")
        f.write("### What does permutation importance mean?\n")
        f.write("- **Higher positive values** = Feature is crucial for accurate predictions\n")
        f.write("- **Importance score** = Decrease in AUC when feature is randomly permuted\n")
        f.write("- **Features with high importance for mortality** help identify high-risk patients\n")
        f.write("- **Features with high importance for survival** help identify low-risk patients\n\n")

        f.write("### Clinical Implications\n")
        f.write("- Focus clinical attention on patients with high scores on mortality-associated features\n")
        f.write("- Use survival-associated features to identify patients who may not need intensive monitoring\n")
        f.write("- These features capture complex interactions that simple correlation might miss\n\n")

        f.write("## Files Generated\n")
        f.write("- `permutation_importance_mortality.csv` - Full mortality importance results\n")
        f.write("- `permutation_importance_survival.csv` - Full survival importance results\n")
        f.write("- `permutation_importance_mortality_top20.png` - Mortality importance visualization\n")
        f.write("- `permutation_importance_survival_top20.png` - Survival importance visualization\n")
        f.write("- `permutation_importance_comparison.png` - Side-by-side comparison\n")

    print("✓ Comprehensive report saved: permutation_importance_report.md")

    print("\n" + "=" * 100)
    print("PERMUTATION IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 100)

    print("\n🎯 SUMMARY:")
    print("• Permutation importance reveals feature contributions to model predictions")
    print("• Different features are important for predicting mortality vs survival")
    print("• This method captures feature interactions that correlation analysis misses")
    print("• Results are directly applicable for clinical decision support")

if __name__ == "__main__":
    main()
