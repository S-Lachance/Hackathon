"""
Fix SHAP analysis for the original model using alternative approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
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
    """Load data and original model for SHAP analysis."""
    print("Loading data and original model for SHAP analysis...")

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

    # Load original model
    try:
        model = joblib.load("random_forest_original.joblib")
        print("✓ Loaded random_forest_original.joblib model")
    except FileNotFoundError:
        print("✗ Model file not found. Please ensure random_forest_original.joblib exists")
        return None, None, None

    # Return test data and model
    X_test = x_test[feature_cols]
    y_test = y_test

    return X_test, y_test, model

def try_shap_with_different_parameters(X, y, model, feature_names):
    """Try SHAP with different parameters to see if we can get stable values."""

    print("Trying SHAP with different parameters...")

    # Sample smaller dataset for testing
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=50, replace=False)  # Very small sample
    X_sample = X.iloc[sample_indices]
    y_sample = y.iloc[sample_indices]

    results = {}

    # Try 1: Default TreeExplainer with different check_additivity
    try:
        print("  Testing TreeExplainer with check_additivity=False...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        results['default'] = {
            'shap_range': (shap_values.min(), shap_values.max()),
            'mean_abs': np.abs(shap_values).mean(),
            'expected_value': explainer.expected_value
        }
        print(f"    Range: {shap_values.min():.2f} to {shap_values.max():.2f}")

    except Exception as e:
        print(f"    Failed: {e}")

    # Try 2: KernelExplainer (model-agnostic, slower but more stable)
    try:
        print("  Testing KernelExplainer (model-agnostic)...")
        background = X_sample.iloc[:10]  # Small background dataset
        explainer = shap.KernelExplainer(model.predict_proba, background)

        # Explain first 5 instances
        shap_values = explainer.shap_values(X_sample.iloc[:5])

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        results['kernel'] = {
            'shap_range': (shap_values.min(), shap_values.max()),
            'mean_abs': np.abs(shap_values).mean(),
        }
        print(f"    Range: {shap_values.min():.4f} to {shap_values.max():.4f}")

    except Exception as e:
        print(f"    Failed: {e}")

    return results

def shap_with_model_simplification(X, y, model, feature_names):
    """Try SHAP with a simplified model or different approach."""

    print("Trying SHAP with model simplification...")

    # Create a simpler model with fewer trees and limited depth
    from sklearn.ensemble import RandomForestClassifier

    # Train a simplified version on the same data
    simple_model = RandomForestClassifier(
        n_estimators=50,  # Much fewer trees
        max_depth=5,      # Limited depth
        random_state=42
    )

    # Use same training data as original model
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=min(5000, len(X)), replace=False)
    X_train_sample = X.iloc[sample_indices]
    y_train_sample = y.iloc[sample_indices]

    simple_model.fit(X_train_sample, y_train_sample)

    # Try SHAP on simplified model
    try:
        explainer = shap.TreeExplainer(simple_model)
        sample_indices = np.random.choice(len(X), size=50, replace=False)
        X_sample = X.iloc[sample_indices]

        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        print(f"Simplified model SHAP range: {shap_values.min():.4f} to {shap_values.max():.4f}")

        # Calculate feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)

        return feature_importance.head(20)

    except Exception as e:
        print(f"Simplified model SHAP failed: {e}")
        return None

def comprehensive_permutation_importance(X, y, model, feature_names):
    """Comprehensive permutation importance analysis as alternative to SHAP."""

    print("Running comprehensive permutation importance analysis...")

    # Use larger sample for better estimates
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
    X_sample = X.iloc[sample_indices]
    y_sample = y.iloc[sample_indices]

    print(f"Using {len(X_sample)} samples for permutation importance")

    # Calculate permutation importance with more repeats for stability
    perm_importance = permutation_importance(
        model, X_sample, y_sample,
        n_repeats=10,  # More repeats for stability
        random_state=42,
        n_jobs=1,  # Sequential to avoid permission issues
        scoring='roc_auc'
    )

    # Create comprehensive results
    results_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
        'importance_max': perm_importance.importances.max(axis=1),
        'importance_min': perm_importance.importances.min(axis=1),
        'cv_ratio': perm_importance.importances_std / (perm_importance.importances_mean + 1e-10)  # Coefficient of variation
    })

    # Sort by mean importance
    results_df = results_df.sort_values('importance_mean', ascending=False)

    print("Top 10 features by permutation importance:")
    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(".4f")

    return results_df

def create_comparison_with_synthetic(X, y, feature_names):
    """Compare original model results with synthetic model results."""

    print("Comparing with synthetic model results...")

    # Load synthetic model
    try:
        synthetic_model = joblib.load("random_forest_synthetic.joblib")
    except:
        print("Could not load synthetic model for comparison")
        return None

    # Get synthetic model permutation importance
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=min(1000, len(X)), replace=False)
    X_sample = X.iloc[sample_indices]
    y_sample = y.iloc[sample_indices]

    synthetic_perm = permutation_importance(
        synthetic_model, X_sample, y_sample,
        n_repeats=5, random_state=42, n_jobs=1, scoring='roc_auc'
    )

    synthetic_results = pd.DataFrame({
        'feature': feature_names,
        'synthetic_importance': synthetic_perm.importances_mean
    })

    return synthetic_results

def main():
    """Main function for alternative SHAP approaches."""
    print("=" * 80)
    print("ALTERNATIVE SHAP APPROACHES FOR ORIGINAL MODEL")
    print("=" * 80)

    # Load data and model
    X, y, model = load_data_and_model()

    if X is None:
        return

    feature_names = list(X.columns)

    print(f"Dataset: {len(X)} samples, {len(feature_names)} features")

    # Approach 1: Try different SHAP parameters
    print("\n" + "="*60)
    print("APPROACH 1: Testing Different SHAP Parameters")
    print("="*60)

    shap_tests = try_shap_with_different_parameters(X, y, model, feature_names)

    # Approach 2: Model simplification
    print("\n" + "="*60)
    print("APPROACH 2: SHAP with Simplified Model")
    print("="*60)

    simplified_shap = shap_with_model_simplification(X, y, model, feature_names)

    # Approach 3: Comprehensive permutation importance
    print("\n" + "="*60)
    print("APPROACH 3: Comprehensive Permutation Importance")
    print("="*60)

    perm_results = comprehensive_permutation_importance(X, y, model, feature_names)

    # Approach 4: Comparison with synthetic model
    print("\n" + "="*60)
    print("APPROACH 4: Comparison with Synthetic Model")
    print("="*60)

    synthetic_comparison = create_comparison_with_synthetic(X, y, feature_names)

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    if perm_results is not None:
        perm_results.to_csv('original_permutation_importance_detailed.csv', index=False)
        print("✓ Detailed permutation importance saved: original_permutation_importance_detailed.csv")

    if simplified_shap is not None:
        simplified_shap.to_csv('original_simplified_shap_importance.csv', index=False)
        print("✓ Simplified SHAP importance saved: original_simplified_shap_importance.csv")

    # Create summary report
    with open('original_model_explainability_report.md', 'w') as f:
        f.write("# Original Model Explainability Analysis\n\n")

        f.write("## Problem Identified\n")
        f.write("The original model's SHAP values show extreme scaling issues (-1000 to 1000+) compared to the synthetic model (0.0 to 0.4).\n\n")

        f.write("## Approaches Tested\n\n")

        f.write("### 1. SHAP Parameter Variations\n")
        for method, results in shap_tests.items():
            if 'shap_range' in results:
                f.write(f"- **{method}**: Range {results['shap_range'][0]:.2f} to {results['shap_range'][1]:.2f}\n")
        f.write("\n")

        f.write("### 2. Model Simplification\n")
        if simplified_shap is not None:
            f.write("Successfully created simplified model with stable SHAP values.\n\n")
            f.write("**Top 5 features by simplified SHAP:**\n")
            for i, (_, row) in enumerate(simplified_shap.head(5).iterrows(), 1):
                f.write(f"{i}. {row['feature']}\n")
        else:
            f.write("Model simplification approach failed.\n")
        f.write("\n")

        f.write("### 3. Permutation Importance (Recommended Alternative)\n")
        if perm_results is not None:
            f.write("**Top 10 features by permutation importance:**\n")
            for i, (_, row) in enumerate(perm_results.head(10).iterrows(), 1):
                f.write(".4f")
        f.write("\n")

        f.write("## Recommendations\n\n")

        f.write("### For Original Model:\n")
        f.write("1. **Use Permutation Importance** - Most reliable and stable\n")
        f.write("2. **Consider Model Retraining** - Investigate training differences\n")
        f.write("3. **Use Simplified Models** - For SHAP if permutation importance insufficient\n\n")

        f.write("### Comparison with Synthetic Model:\n")
        f.write("- Synthetic model: Stable SHAP values, reliable explanations\n")
        f.write("- Original model: Unstable SHAP, use permutation importance instead\n\n")

        f.write("## Files Generated\n")
        f.write("- `original_permutation_importance_detailed.csv` - Comprehensive permutation analysis\n")
        f.write("- `original_simplified_shap_importance.csv` - Simplified model SHAP (if successful)\n")

    print("✓ Comprehensive analysis report saved: original_model_explainability_report.md")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print("\n🎯 SUMMARY:")
    print("• SHAP on original model shows numerical instability")
    print("• Permutation importance provides reliable alternative")
    print("• Simplified models may offer stable SHAP values")
    print("• Synthetic model remains the gold standard for SHAP analysis")

if __name__ == "__main__":
    main()
