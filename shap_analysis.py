"""
SHAP (SHapley Additive exPlanations) Analysis for Mortality Prediction
Identify features that contribute most to oym=TRUE and oym=FALSE predictions
using SHAP values from the trained Random Forest model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import re
import joblib
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def get_feature_columns(df: pd.DataFrame) -> tuple:
    """Returns predictors to use for the POYM task."""
    DX_COLS = [dx for dx in list(df.columns) if re.compile("^(dx_).+").search(dx)]
    ADM_COLS = [adm for adm in list(df.columns) if re.compile("^(adm_).+").search(adm)]
    CAT_COLS = ['gender', 'living_status', 'admission_group', 'service_group']
    CONT_COLS = ["age_original", "ed_visit_count", "ho_ambulance_count", "total_duration"]
    OTHER_BIN_COLS = ["flu_season", "is_ambulance", "is_icu_start_ho", "is_urg_readm", "has_dx"]
    BIN_COLS = DX_COLS + ADM_COLS + OTHER_BIN_COLS
    OYM = "oym"
    return CONT_COLS, BIN_COLS, CAT_COLS, OYM

def load_data_and_model():
    """Load data and trained model for SHAP analysis."""
    print("Loading data and model for SHAP analysis...")

    # Load data exactly as in evaluate_baselines.py
    data = pd.read_csv('csv/dataset.csv')
    print(f"✓ Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")

    # Get column specifications
    cont_cols, bin_cols, cat_cols, target = get_feature_columns(data)

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

    # Load trained model
    try:
        model = joblib.load("random_forest_synthetic.joblib")
        print("✓ Loaded random_forest_synthetic.joblib model")
    except FileNotFoundError:
        print("✗ Model file not found. Please ensure random_forest_synthetic.joblib exists")
        return None, None, None

    # Return test data and model
    X_test = x_test[feature_cols]
    y_test = y_test

    return X_test, y_test, model

def calculate_shap_values(model, X, sample_size=1000):
    """
    Calculate SHAP values for a sample of instances.

    Parameters:
    - model: Trained model
    - X: Feature matrix
    - sample_size: Number of instances to sample for SHAP calculation

    Returns:
    - shap_values: SHAP values array
    - X_sample: Sampled feature matrix
    - y_sample: Corresponding target values
    """
    print(f"Calculating SHAP values for {sample_size} sample instances...")

    # Sample instances for SHAP calculation (computationally expensive)
    if len(X) > sample_size:
        np.random.seed(42)  # Set seed for reproducibility
        sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = X.index[sample_indices]  # We'll need to get y later
    else:
        X_sample = X
        sample_size = len(X)

    print(f"Using {sample_size} instances for SHAP calculation")

    # Create SHAP explainer with additivity check disabled
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values with additivity check disabled
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # For binary classification, shap_values has shape (n_samples, n_features, n_classes)
    # We want the positive class (oym=TRUE) which is index 1
    if len(shap_values.shape) == 3:
        shap_values_positive = shap_values[:, :, 1]  # SHAP values for positive class
    else:
        shap_values_positive = shap_values

    print(f"✓ Calculated SHAP values: shape {shap_values_positive.shape}")

    return shap_values_positive, X_sample, explainer

def analyze_shap_by_outcome(shap_values, X_sample, y_sample, feature_names, outcome_name, top_n=20):
    """
    Analyze SHAP values for instances with specific outcomes.

    Parameters:
    - shap_values: SHAP values array
    - X_sample: Feature matrix sample
    - y_sample: Target values (not used since we sample from test set)
    - feature_names: Feature names
    - outcome_name: "mortality" or "survival"
    - top_n: Number of top features to return

    Returns:
    - top_features_df: DataFrame with top contributing features
    """
    print(f"\nAnalyzing features contributing to {outcome_name}...")

    # Get instances that were predicted as positive (high mortality risk)
    # Since we don't have the actual predictions, we'll analyze all instances
    # and focus on the most positive/negative SHAP values

    # Calculate mean absolute SHAP value for each feature across all instances
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0)
    })

    # Sort by absolute SHAP value (overall importance)
    feature_importance = feature_importance.sort_values('mean_abs_shap', ascending=False)

    print(f"Top {top_n} features by mean absolute SHAP value:")
    top_features = feature_importance.head(top_n)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        direction = "↑" if row['mean_shap'] > 0 else "↓"
        print("2d")
    # Also analyze features that most contribute to positive predictions (mortality)
    # vs negative predictions (survival)

    if outcome_name == "mortality":
        # For mortality: features with most positive SHAP values
        shap_mortality = shap_values[shap_values.sum(axis=1) > np.percentile(shap_values.sum(axis=1), 75)]
        if len(shap_mortality) > 0:
            mortality_features = pd.DataFrame({
                'feature': feature_names,
                'mean_shap_mortality': shap_mortality.mean(axis=0)
            }).sort_values('mean_shap_mortality', ascending=False)
            print(f"\nFeatures most contributing to HIGH mortality predictions:")
            for i, (_, row) in enumerate(mortality_features.head(10).iterrows(), 1):
                print("2d")
    elif outcome_name == "survival":
        # For survival: features with most negative SHAP values (pushing toward survival)
        shap_survival = shap_values[shap_values.sum(axis=1) < np.percentile(shap_values.sum(axis=1), 25)]
        if len(shap_survival) > 0:
            survival_features = pd.DataFrame({
                'feature': feature_names,
                'mean_shap_survival': shap_survival.mean(axis=0)
            }).sort_values('mean_shap_survival', ascending=True)  # Most negative first
            print(f"\nFeatures most contributing to LOW mortality predictions (survival):")
            for i, (_, row) in enumerate(survival_features.head(10).iterrows(), 1):
                print("2d")
    return feature_importance

def create_shap_visualizations(shap_values, X_sample, feature_names, sample_size=100):
    """Create SHAP visualizations."""
    print("\nCreating SHAP visualizations...")

    # Sample smaller subset for clearer plots
    if len(X_sample) > sample_size:
        np.random.seed(42)  # Set seed for reproducibility
        sample_indices = np.random.choice(len(X_sample), size=sample_size, replace=False)
        shap_sample = shap_values[sample_indices]
        X_sample_small = X_sample.iloc[sample_indices]
    else:
        shap_sample = shap_values
        X_sample_small = X_sample

    # 1. SHAP Summary Plot (Top features)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_sample, X_sample_small, feature_names=feature_names,
                     max_display=20, show=False)
    plt.title("SHAP Summary Plot: Feature Importance for Mortality Prediction")
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ SHAP summary plot saved: shap_summary_plot.png")

    # 2. SHAP Bar Plot (Mean absolute SHAP values)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_sample, X_sample_small, feature_names=feature_names,
                     plot_type="bar", max_display=20, show=False)
    plt.title("SHAP Feature Importance: Mean Absolute SHAP Values")
    plt.tight_layout()
    plt.savefig('shap_bar_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ SHAP bar plot saved: shap_bar_plot.png")

    # 3. Waterfall plot for a single instance (high mortality risk)
    high_risk_idx = np.argmax(shap_sample.sum(axis=1))  # Instance with highest predicted mortality
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(values=shap_sample[high_risk_idx],
                                         base_values=0,  # This would be the model's expected value
                                         data=X_sample_small.iloc[high_risk_idx],
                                         feature_names=feature_names), show=False)
    plt.title("SHAP Waterfall Plot: High Mortality Risk Instance")
    plt.tight_layout()
    plt.savefig('shap_waterfall_high_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ SHAP waterfall plot saved: shap_waterfall_high_risk.png")

    # 4. Waterfall plot for a single instance (low mortality risk)
    low_risk_idx = np.argmin(shap_sample.sum(axis=1))  # Instance with lowest predicted mortality
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(values=shap_sample[low_risk_idx],
                                         base_values=0,
                                         data=X_sample_small.iloc[low_risk_idx],
                                         feature_names=feature_names), show=False)
    plt.title("SHAP Waterfall Plot: Low Mortality Risk Instance")
    plt.tight_layout()
    plt.savefig('shap_waterfall_low_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ SHAP waterfall plot saved: shap_waterfall_low_risk.png")

def main():
    """Main function for SHAP analysis."""
    print("=" * 80)
    print("SHAP (SHapley Additive exPlanations) ANALYSIS")
    print("Identifying features contributing to oym=TRUE and oym=FALSE")
    print("=" * 80)

    # Load data and model
    X_test, y_test, model = load_data_and_model()

    if X_test is None:
        return

    feature_names = list(X_test.columns)

    # Calculate SHAP values (sample for computational efficiency)
    sample_size = min(1000, len(X_test))  # Use up to 1000 instances
    shap_values, X_sample, explainer = calculate_shap_values(model, X_test, sample_size)

    # Analyze features for mortality (oym=TRUE)
    mortality_features = analyze_shap_by_outcome(
        shap_values, X_sample, None, feature_names, "mortality", top_n=20
    )

    # Analyze features for survival (oym=FALSE)
    survival_features = analyze_shap_by_outcome(
        shap_values, X_sample, None, feature_names, "survival", top_n=20
    )

    # Create visualizations
    create_shap_visualizations(shap_values, X_sample, feature_names)

    # Save results
    print("\nSaving SHAP analysis results...")
    mortality_features.to_csv('shap_mortality_features.csv', index=False)
    survival_features.to_csv('shap_survival_features.csv', index=False)

    print("✓ SHAP mortality features saved: shap_mortality_features.csv")
    print("✓ SHAP survival features saved: shap_survival_features.csv")

    # Create summary report
    with open('shap_analysis_report.md', 'w') as f:
        f.write("# SHAP Analysis Report: Feature Contributions to Mortality\n\n")

        f.write("## Overview\n")
        f.write("SHAP (SHapley Additive exPlanations) analysis quantifies how much each feature contributes to the model's prediction for each individual instance.\n\n")
        f.write("**Method**: TreeExplainer on Random Forest model\n")
        f.write(f"**Sample size**: {len(X_sample)} instances analyzed\n\n")

        f.write("## Top 20 Features by Mean Absolute SHAP Value\n")
        f.write("Features with largest overall impact on predictions:\n\n")
        top_overall = mortality_features.head(20)
        f.write("| Rank | Feature | Mean |SHAP| | Direction | Std Dev |\n")
        f.write("|------|---------|--------|----------|--------|\n")
        for i, (_, row) in enumerate(top_overall.iterrows(), 1):
            direction = "↑ Risk" if row['mean_shap'] > 0 else "↓ Protective"
            f.write("2d")

        f.write("\n## Top 10 Features Contributing to Mortality (oym=TRUE)\n")
        f.write("Features that most increase predicted mortality risk:\n\n")
        f.write("| Rank | Feature | Mean SHAP |\n")
        f.write("|------|---------|-----------|\n")
        # Calculate mortality-specific features
        mortality_specific = mortality_features.nlargest(10, 'mean_shap')
        for i, (_, row) in enumerate(mortality_specific.iterrows(), 1):
            f.write("2d")

        f.write("\n## Top 10 Features Contributing to Survival (oym=FALSE)\n")
        f.write("Features that most decrease predicted mortality risk:\n\n")
        f.write("| Rank | Feature | Mean SHAP |\n")
        f.write("|------|---------|-----------|\n")
        # Calculate survival-specific features
        survival_specific = survival_features.nsmallest(10, 'mean_shap')
        for i, (_, row) in enumerate(survival_specific.iterrows(), 1):
            f.write("2d")

        f.write("\n## Interpretation\n")
        f.write("### SHAP Values Meaning:\n")
        f.write("- **Positive SHAP**: Feature pushes prediction toward mortality (oym=TRUE)\n")
        f.write("- **Negative SHAP**: Feature pushes prediction toward survival (oym=FALSE)\n")
        f.write("- **Magnitude**: Strength of the feature's contribution\n\n")

        f.write("### Clinical Applications:\n")
        f.write("- **Risk factors**: Features with consistently positive SHAP values\n")
        f.write("- **Protective factors**: Features with consistently negative SHAP values\n")
        f.write("- **Individual predictions**: SHAP explains why each patient receives their risk score\n\n")

        f.write("## Files Generated\n")
        f.write("- `shap_mortality_features.csv` - Full SHAP analysis for mortality\n")
        f.write("- `shap_survival_features.csv` - Full SHAP analysis for survival\n")
        f.write("- `shap_summary_plot.png` - SHAP summary visualization\n")
        f.write("- `shap_bar_plot.png` - Feature importance bar chart\n")
        f.write("- `shap_waterfall_high_risk.png` - Example high-risk prediction explanation\n")
        f.write("- `shap_waterfall_low_risk.png` - Example low-risk prediction explanation\n")

    print("✓ Comprehensive SHAP report saved: shap_analysis_report.md")

    print("\n" + "=" * 80)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 80)

    print("\n🎯 SUMMARY:")
    print("• SHAP quantifies each feature's contribution to individual predictions")
    print("• Positive SHAP values increase mortality risk prediction")
    print("• Negative SHAP values decrease mortality risk prediction")
    print("• Results show both global feature importance and individual explanations")
    print("• Provides model interpretability for clinical decision support")

if __name__ == "__main__":
    main()
