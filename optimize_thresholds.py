"""
Threshold Optimization for Clinical Decision Support
Finds optimal operating points for 1:1 and 1:10 alert scenarios
"""

import numpy as np
import pandas as pd
import joblib
import re
from typing import Tuple, List
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score
)
import matplotlib.pyplot as plt

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
    """Calculate all metrics at a specific threshold"""
    y_pred = (y_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Alert ratio: FP / TP (false alerts per true alert)
    alert_ratio = fp / tp if tp > 0 else float('inf')
    
    # Patients flagged
    patients_flagged = tp + fp
    patients_flagged_pct = patients_flagged / len(y_true) * 100
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'alert_ratio': alert_ratio,
        'patients_flagged': int(patients_flagged),
        'patients_flagged_pct': patients_flagged_pct
    }

def find_optimal_thresholds(y_true, y_proba):
    """Find optimal thresholds for different scenarios"""
    
    # Test many thresholds
    thresholds = np.linspace(0, 1, 1001)
    results = []
    
    for thresh in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_proba, thresh)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Scenario 1: 1:1 ratio (1 false alert per true alert)
    # Find threshold closest to alert_ratio = 1.0
    df_valid = df[df['tp'] > 0]  # Only consider thresholds that catch someone
    df_1to1 = df_valid.copy()
    df_1to1['distance_from_1'] = abs(df_1to1['alert_ratio'] - 1.0)
    optimal_1to1 = df_1to1.loc[df_1to1['distance_from_1'].idxmin()]
    
    # Scenario 2: 1:10 ratio (10 false alerts per true alert - high recall)
    df_1to10 = df_valid.copy()
    df_1to10['distance_from_10'] = abs(df_1to10['alert_ratio'] - 10.0)
    optimal_1to10 = df_1to10.loc[df_1to10['distance_from_10'].idxmin()]
    
    # Scenario 3: Max F1-Score (balanced)
    optimal_f1 = df.loc[df['f1'].idxmax()]
    
    # Scenario 4: Youden's Index (max sensitivity + specificity - 1)
    df['youden'] = df['recall'] + df['specificity'] - 1
    optimal_youden = df.loc[df['youden'].idxmax()]
    
    return {
        '1:1 Alert Ratio': optimal_1to1,
        '1:10 Alert Ratio': optimal_1to10,
        'Max F1-Score': optimal_f1,
        'Max Youden Index': optimal_youden
    }, df

def print_scenario_report(scenario_name, metrics):
    """Print a formatted report for a specific scenario"""
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    print(f"Decision Threshold:     {metrics['threshold']:.4f}")
    print(f"\nClinical Impact:")
    print(f"  Patients Flagged:     {metrics['patients_flagged']:,} ({metrics['patients_flagged_pct']:.2f}% of population)")
    print(f"  Alert Ratio:          {metrics['alert_ratio']:.2f}:1 (FP:TP)")
    print(f"  True Alerts (TP):     {metrics['tp']:,}")
    print(f"  False Alerts (FP):    {metrics['fp']:,}")
    print(f"\nPerformance Metrics:")
    print(f"  Recall (Sensitivity): {metrics['recall']:.4f} ({metrics['tp']} of {metrics['tp'] + metrics['fn']} high-risk patients caught)")
    print(f"  Precision:            {metrics['precision']:.4f}")
    print(f"  Specificity:          {metrics['specificity']:.4f}")
    print(f"  F1-Score:             {metrics['f1']:.4f}")
    print(f"  Accuracy:             {metrics['accuracy']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']:>6,}  FP: {metrics['fp']:>6,}")
    print(f"  FN: {metrics['fn']:>6,}  TP: {metrics['tp']:>6,}")

def plot_threshold_analysis(df, optimal_scenarios, model_name):
    """Create visualization of threshold analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Threshold Optimization Analysis - {model_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Precision-Recall vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(df['threshold'], df['precision'], label='Precision', linewidth=2)
    ax1.plot(df['threshold'], df['recall'], label='Recall', linewidth=2)
    ax1.plot(df['threshold'], df['f1'], label='F1-Score', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision-Recall Trade-off')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    
    # Plot 2: Alert Ratio vs Threshold
    ax2 = axes[0, 1]
    # Filter out inf values for plotting
    df_plot = df[df['alert_ratio'] < 50].copy()
    ax2.plot(df_plot['threshold'], df_plot['alert_ratio'], linewidth=2, color='red')
    ax2.axhline(y=1, color='green', linestyle='--', label='1:1 Target', linewidth=2)
    ax2.axhline(y=10, color='orange', linestyle='--', label='1:10 Target', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('False Alerts per True Alert')
    ax2.set_title('Alert Ratio vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 50])
    
    # Plot 3: Patients Flagged vs Recall
    ax3 = axes[1, 0]
    ax3.plot(df['patients_flagged_pct'], df['recall'], linewidth=2)
    ax3.set_xlabel('% of Patients Flagged')
    ax3.set_ylabel('Recall (High-Risk Patients Caught)')
    ax3.set_title('Resource Impact vs Clinical Effectiveness')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mark optimal points
    ax4 = axes[1, 1]
    # ROC-style but with threshold
    ax4.plot(1 - df['specificity'], df['recall'], linewidth=2, label='Model Performance')
    
    # Mark optimal scenarios
    colors = {'1:1 Alert Ratio': 'green', '1:10 Alert Ratio': 'orange', 
              'Max F1-Score': 'blue', 'Max Youden Index': 'purple'}
    for name, metrics in optimal_scenarios.items():
        fpr = 1 - metrics['specificity']
        ax4.scatter(fpr, metrics['recall'], s=200, color=colors[name], 
                   label=f"{name} (t={metrics['threshold']:.3f})", 
                   edgecolors='black', linewidth=2, zorder=5)
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    ax4.set_xlabel('False Positive Rate (1 - Specificity)')
    ax4.set_ylabel('True Positive Rate (Recall)')
    ax4.set_title('ROC Space with Optimal Operating Points')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'threshold_optimization_{model_name}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: threshold_optimization_{model_name}.png")

def main():
    """Main evaluation pipeline"""
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION FOR CLINICAL DECISION SUPPORT")
    print("RSN Hackathon Challenge: Hospital One-year Mortality Risk Prediction")
    print("="*80)
    
    # Load dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    data = pd.read_csv('csv/dataset.csv')
    print(f"✓ Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Get column specifications
    cont_cols, bin_cols, cat_cols, target = get_cols(data)
    
    # Select one visit per patient
    data = data.groupby("patient_id", group_keys=False).apply(
        lambda x: x.sample(1, random_state=42)
    ).reset_index(drop=True)
    print(f"✓ Selected one visit per patient: {data.shape[0]} rows")
    
    # One hot encoding
    onehot_data = pd.get_dummies(data, columns=cat_cols, dtype=int)
    onehot_cat_cols = [c for c in onehot_data.columns if c not in data.columns]
    
    # Split to test set (same as evaluation)
    np.random.seed(42)
    test_ids = np.random.choice(onehot_data.index, size=int(0.5*len(onehot_data)), replace=False)
    test_set = onehot_data.loc[test_ids]
    
    # Prepare features
    feature_cols = cont_cols + bin_cols + onehot_cat_cols
    X_test = test_set[feature_cols].values
    y_test = test_set[target].values
    
    print(f"✓ Test set: {len(test_set)} samples")
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Positive class: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # Evaluate both models
    models = {
        'Model_1_Synthetic': 'random_forest_synthetic.joblib',
        'Model_2_Original': 'random_forest_original.joblib'
    }
    
    for model_name, model_path in models.items():
        print("\n" + "="*80)
        print(f"OPTIMIZING THRESHOLDS: {model_name.replace('_', ' ')}")
        print("="*80)
        
        # Load model
        model = joblib.load(model_path)
        
        # Get probability predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Find optimal thresholds
        optimal_scenarios, all_results = find_optimal_thresholds(y_test, y_proba)
        
        # Print reports for each scenario
        for scenario_name, metrics in optimal_scenarios.items():
            print_scenario_report(scenario_name, metrics)
        
        # Create visualizations
        plot_threshold_analysis(all_results, optimal_scenarios, model_name)
        
        # Save detailed results
        all_results.to_csv(f'threshold_analysis_{model_name}.csv', index=False)
        print(f"✓ Detailed results saved: threshold_analysis_{model_name}.csv")
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("="*80)
    print("\n📊 Key Takeaways:")
    print("  1. Check the PNG visualizations to understand trade-offs")
    print("  2. Review the 1:1 and 1:10 scenarios for clinical decision-making")
    print("  3. Consider which alert ratio is acceptable for your clinical setting")
    print("  4. Use the optimized thresholds in your final model deployment")
    print("\n")

if __name__ == "__main__":
    main()

