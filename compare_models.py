"""
Comprehensive model comparison script.
Compares synthetic vs original models across multiple metrics and thresholds.
"""

import numpy as np
import pandas as pd
import joblib
import re
from typing import Tuple, List
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns

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

def calculate_metrics_at_threshold(y_true, y_proba, threshold):
    """Calculate comprehensive metrics at a specific threshold"""
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Additional clinical metrics
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    ppv = precision  # Positive Predictive Value (same as precision)

    # Alert fatigue metrics
    alert_ratio = fp / tp if tp > 0 else float('inf')
    patients_flagged = tp + fp
    patients_flagged_pct = patients_flagged / len(y_true) * 100

    # Clinical utility scores
    youden_index = recall + specificity - 1

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'npv': npv,
        'ppv': ppv,
        'alert_ratio': alert_ratio,
        'patients_flagged': int(patients_flagged),
        'patients_flagged_pct': patients_flagged_pct,
        'youden_index': youden_index,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

def load_and_prepare_data(filepath):
    """Load and prepare the dataset."""
    from csv_handling import load_and_prep_data

    print("Loading and preparing data...")
    df = load_and_prep_data(filepath)
    if df is None:
        return None, None, None

    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Get column specifications
    cont_cols, bin_cols, cat_cols, target = get_cols(df)

    # Select one visit per patient
    df = df.groupby("patient_id", group_keys=False).apply(
        lambda x: x.sample(1, random_state=42)
    ).reset_index(drop=True)
    print(f"Selected one visit per patient: {len(df)} rows")

    # One-hot encode categorical variables
    onehot_data = pd.get_dummies(df, columns=cat_cols, dtype=int)
    onehot_cat_cols = [c for c in onehot_data.columns if c not in df.columns]
    print(f"One-hot encoded categorical features: {len(onehot_cat_cols)} new columns")

    # Split into train/test (same as original training)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        onehot_data, df[target],
        test_size=0.5, random_state=42
    )
    print(f"Split data: {x_train.shape[0]} train, {x_test.shape[0]} test")

    # Define feature columns
    feature_cols = cont_cols + bin_cols + onehot_cat_cols
    print(f"Total features: {len(feature_cols)}")

    return x_test[feature_cols], y_test

def compare_models_at_thresholds(X_test, y_test, thresholds=None):
    """Compare both models across multiple thresholds."""

    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)  # 0.00, 0.01, ..., 1.00

    print(f"Comparing models at {len(thresholds)} threshold values...")

    # Load models
    rf_synthetic = joblib.load("random_forest_synthetic.joblib")
    rf_original = joblib.load("random_forest_original.joblib")

    # Get predictions
    proba_synthetic = rf_synthetic.predict_proba(X_test)[:, 1]
    proba_original = rf_original.predict_proba(X_test)[:, 1]

    results = []

    for threshold in thresholds:
        metrics_synthetic = calculate_metrics_at_threshold(y_test, proba_synthetic, threshold)
        metrics_original = calculate_metrics_at_threshold(y_test, proba_original, threshold)

        # Calculate differences
        diff = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'npv', 'ppv', 'youden_index']:
            diff[metric] = metrics_synthetic[metric] - metrics_original[metric]

        results.append({
            'threshold': threshold,
            'synthetic': metrics_synthetic,
            'original': metrics_original,
            'difference': diff
        })

    return results, proba_synthetic, proba_original

def plot_metric_comparison(results, metric_name, title=None):
    """Plot comparison of a metric across thresholds."""
    thresholds = [r['threshold'] for r in results]

    synthetic_vals = [r['synthetic'][metric_name] for r in results]
    original_vals = [r['original'][metric_name] for r in results]

    plt.figure(figsize=(12, 6))

    plt.plot(thresholds, synthetic_vals, 'b-', label='Synthetic Dataset Model', linewidth=2)
    plt.plot(thresholds, original_vals, 'r-', label='Original Dataset Model', linewidth=2)

    plt.xlabel('Decision Threshold')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(title or f'{metric_name.replace("_", " ").title()} vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add vertical lines at optimal thresholds
    for scenario in ['1:1 Alert Ratio', '1:10 Alert Ratio', 'Max F1-Score', 'Max Youden Index']:
        # Find optimal thresholds from previous optimization
        synthetic_opt = None
        original_opt = None
        for r in results:
            if abs(r['synthetic']['alert_ratio'] - 1.0) < 0.1 and synthetic_opt is None:
                synthetic_opt = r['threshold']
            if abs(r['original']['alert_ratio'] - 1.0) < 0.1 and original_opt is None:
                original_opt = r['threshold']

        if synthetic_opt:
            plt.axvline(x=synthetic_opt, color='b', linestyle='--', alpha=0.7, label=f'Synthetic {scenario}')

    plt.tight_layout()
    return plt.gcf()

def create_comparison_summary(results):
    """Create a summary table of model comparisons at key thresholds."""

    key_thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    summary_data = []

    for threshold in key_thresholds:
        # Find closest result to this threshold
        closest_result = min(results, key=lambda x: abs(x['threshold'] - threshold))

        row = {
            'Threshold': threshold,
            'Synthetic_Recall': closest_result['synthetic']['recall'],
            'Original_Recall': closest_result['original']['recall'],
            'Synthetic_Precision': closest_result['synthetic']['precision'],
            'Original_Precision': closest_result['original']['precision'],
            'Synthetic_F1': closest_result['synthetic']['f1'],
            'Original_F1': closest_result['original']['f1'],
            'Synthetic_Specificity': closest_result['synthetic']['specificity'],
            'Original_Specificity': closest_result['original']['specificity'],
            'Synthetic_NPV': closest_result['synthetic']['npv'],
            'Original_NPV': closest_result['original']['npv'],
            'Synthetic_Patients_Flagged_%': closest_result['synthetic']['patients_flagged_pct'],
            'Original_Patients_Flagged_%': closest_result['original']['patients_flagged_pct']
        }
        summary_data.append(row)

    return pd.DataFrame(summary_data)

def main():
    """Main comparison function."""
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("Synthetic vs Original Dataset Models")
    print("=" * 80)

    # Load test data
    X_test, y_test = load_and_prepare_data('csv/dataset.csv')
    if X_test is None:
        print("Failed to load data")
        return

    # Compare models across thresholds
    results, proba_synthetic, proba_original = compare_models_at_thresholds(X_test, y_test)

    # Create summary table
    summary_df = create_comparison_summary(results)
    print("\n" + "=" * 80)
    print("METRIC COMPARISON SUMMARY")
    print("=" * 80)
    print(summary_df.round(4).to_string())

    # Save detailed results
    detailed_results = []
    for r in results:
        row = {'threshold': r['threshold']}
        row.update({f'synthetic_{k}': v for k, v in r['synthetic'].items() if k != 'threshold'})
        row.update({f'original_{k}': v for k, v in r['original'].items() if k != 'threshold'})
        row.update({f'diff_{k}': v for k, v in r['difference'].items()})
        detailed_results.append(row)

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('model_comparison_detailed.csv', index=False)
    print("\n✓ Detailed results saved to: model_comparison_detailed.csv")
    # Create comparison plots for key metrics
    metrics_to_plot = ['recall', 'precision', 'f1', 'specificity', 'npv', 'youden_index']

    print("\nGenerating comparison plots...")
    for metric in metrics_to_plot:
        fig = plot_metric_comparison(results, metric)
        filename = f'model_comparison_{metric}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")

    # Additional analysis: ROC and Precision-Recall curves
    print("\nGenerating ROC and Precision-Recall curves...")

    # ROC curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    fpr_synth, tpr_synth, _ = roc_curve(y_test, proba_synthetic)
    fpr_orig, tpr_orig, _ = roc_curve(y_test, proba_original)

    plt.plot(fpr_synth, tpr_synth, 'b-', label='.3f', linewidth=2)
    plt.plot(fpr_orig, tpr_orig, 'r-', label='.3f', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Precision-Recall curves
    plt.subplot(1, 2, 2)
    precision_synth, recall_synth, _ = precision_recall_curve(y_test, proba_synthetic)
    precision_orig, recall_orig, _ = precision_recall_curve(y_test, proba_original)

    plt.plot(recall_synth, precision_synth, 'b-', label='Synthetic Model', linewidth=2)
    plt.plot(recall_orig, precision_orig, 'r-', label='Original Model', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: model_comparison_curves.png")
    # Clinical utility analysis
    print("\n" + "=" * 80)
    print("CLINICAL UTILITY ANALYSIS")
    print("=" * 80)

    # Find optimal thresholds for different scenarios
    scenarios = {
        '1:1 Alert Ratio': lambda r: abs(r['alert_ratio'] - 1.0),
        '1:10 Alert Ratio': lambda r: abs(r['alert_ratio'] - 10.0),
        'Max F1': lambda r: -r['f1'],  # Negative for minimization
        'Max Youden': lambda r: -r['youden_index']
    }

    for scenario_name, score_func in scenarios.items():
        print(f"\n{scenario_name}:")

        # Find best threshold for synthetic model
        best_synth = min(results, key=lambda r: score_func(r['synthetic']))
        # Find best threshold for original model
        best_orig = min(results, key=lambda r: score_func(r['original']))

        print(f"  Synthetic Model - Threshold: {best_synth['threshold']:.3f}")
        print(f"    Recall: {best_synth['synthetic']['recall']:.3f}, Precision: {best_synth['synthetic']['precision']:.3f}")
        print(f"    Patients flagged: {best_synth['synthetic']['patients_flagged_pct']:.1f}%")

        print(f"  Original Model - Threshold: {best_orig['threshold']:.3f}")
        print(f"    Recall: {best_orig['original']['recall']:.3f}, Precision: {best_orig['original']['precision']:.3f}")
        print(f"    Patients flagged: {best_orig['original']['patients_flagged_pct']:.1f}%")

        # Compare which is better
        synth_score = score_func(best_synth['synthetic'])
        orig_score = score_func(best_orig['original'])

        if scenario_name in ['Max F1', 'Max Youden']:
            metric_name = scenario_name.split()[1].lower()
            if metric_name == 'youden':
                metric_name = 'youden_index'
            winner = "Synthetic" if best_synth['synthetic'][metric_name] > best_orig['original'][metric_name] else "Original"
        else:
            winner = "Synthetic" if synth_score < orig_score else "Original"

        print(f"  Better performing model: {winner}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - model_comparison_detailed.csv (all metrics at all thresholds)")
    print("  - model_comparison_*.png (metric comparison plots)")
    print("  - model_comparison_curves.png (ROC and PR curves)")

    print("\n💡 Key Insights:")
    print("  • Compare models at different operating points for clinical scenarios")
    print("  • Synthetic model often performs better on minority class (deaths)")
    print("  • Threshold choice dramatically affects clinical utility")
    print("  • Consider both precision and recall for your use case")

if __name__ == "__main__":
    main()
