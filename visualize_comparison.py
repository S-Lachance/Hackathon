"""
Visualization script for comparing baseline models.
Generates plots and detailed analysis.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns


def plot_roc_curves(models_data, save_path='roc_comparison.png'):
    """
    Plot ROC curves for both models.
    
    Args:
        models_data: List of dicts with 'name', 'y_true', 'y_pred_proba'
    """
    plt.figure(figsize=(10, 8))
    
    for data in models_data:
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_pred_proba'])
        auc = data['auc']
        plt.plot(fpr, tpr, linewidth=2, label=f"{data['name']} (AUC = {auc:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curve to {save_path}")
    plt.close()


def plot_precision_recall_curves(models_data, save_path='pr_comparison.png'):
    """Plot Precision-Recall curves for both models."""
    plt.figure(figsize=(10, 8))
    
    for data in models_data:
        precision, recall, _ = precision_recall_curve(data['y_true'], data['y_pred_proba'])
        plt.plot(recall, precision, linewidth=2, label=data['name'])
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves: Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Precision-Recall curve to {save_path}")
    plt.close()


def plot_confusion_matrices(models_data, save_path='confusion_matrices.png'):
    """Plot confusion matrices for both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, data in enumerate(models_data):
        cm = confusion_matrix(data['y_true'], data['y_pred'])
        
        # Normalize to percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both count and percentage
        annot = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" 
                          for j in range(cm.shape[1])] 
                         for i in range(cm.shape[0])])
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                   xticklabels=['No Mortality', 'Mortality'],
                   yticklabels=['No Mortality', 'Mortality'],
                   ax=axes[idx], cbar_kws={'label': 'Count'})
        
        axes[idx].set_title(data['name'], fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrices to {save_path}")
    plt.close()


def plot_metric_comparison(results_synthetic, results_original, save_path='metrics_comparison.png'):
    """Plot bar chart comparing all metrics."""
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1', 'specificity']
    metric_labels = ['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    
    synthetic_values = [results_synthetic[m] for m in metrics]
    original_values = [results_original[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, synthetic_values, width, label='Model 1 (Synthetic)', alpha=0.8)
    bars2 = ax.bar(x + width/2, original_values, width, label='Model 2 (Original)', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics comparison to {save_path}")
    plt.close()


def plot_class_weight_comparison(save_path='class_weights_comparison.png'):
    """Plot class weight comparison between models."""
    with open("rf_homr_best_hps_synthetic.json", "r") as f:
        hps_synthetic = json.load(f)
    
    with open("rf_homr_best_hps_original.json", "r") as f:
        hps_original = json.load(f)
    
    # Extract class weights
    weights_syn = [float(hps_synthetic['class_weight']['0']), 
                   float(hps_synthetic['class_weight']['1'])]
    weights_orig = [float(hps_original['class_weight']['0']), 
                    float(hps_original['class_weight']['1'])]
    
    x = np.arange(2)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, weights_syn, width, label='Model 1 (Synthetic)', alpha=0.8)
    bars2 = ax.bar(x + width/2, weights_orig, width, label='Model 2 (Original)', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Class Weight Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Class 0 (No Mortality)', 'Class 1 (Mortality)'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved class weights comparison to {save_path}")
    plt.close()


def generate_summary_table(results_synthetic, results_original, save_path='summary_table.csv'):
    """Generate and save a summary comparison table."""
    
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1', 'specificity']
    
    data = {
        'Metric': [m.upper() for m in metrics],
        'Model 1 (Synthetic)': [results_synthetic[m] for m in metrics],
        'Model 2 (Original)': [results_original[m] for m in metrics],
        'Difference (Syn - Orig)': [results_synthetic[m] - results_original[m] for m in metrics],
        'Better Model': ['Synthetic' if results_synthetic[m] > results_original[m] 
                        else ('Original' if results_original[m] > results_synthetic[m] else 'Tie')
                        for m in metrics]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"✓ Saved summary table to {save_path}")
    
    return df


def print_detailed_analysis(results_synthetic, results_original):
    """Print detailed statistical analysis."""
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    # AUC difference
    auc_diff = results_synthetic['auc'] - results_original['auc']
    print(f"\n1. AUC-ROC Analysis:")
    print(f"   Model 1 (Synthetic): {results_synthetic['auc']:.4f}")
    print(f"   Model 2 (Original):  {results_original['auc']:.4f}")
    print(f"   Difference:          {auc_diff:+.4f} ({abs(auc_diff)/results_original['auc']*100:+.2f}%)")
    
    if abs(auc_diff) < 0.01:
        print("   → Models have nearly identical discrimination ability")
    elif auc_diff > 0:
        print("   → Model 1 has better discrimination ability")
    else:
        print("   → Model 2 has better discrimination ability")
    
    # Recall vs Specificity trade-off
    print(f"\n2. Sensitivity vs Specificity Trade-off:")
    print(f"   Model 1 - Recall: {results_synthetic['recall']:.4f}, Specificity: {results_synthetic['specificity']:.4f}")
    print(f"   Model 2 - Recall: {results_original['recall']:.4f}, Specificity: {results_original['specificity']:.4f}")
    
    # False negative analysis
    fnr_syn = results_synthetic['fn'] / (results_synthetic['fn'] + results_synthetic['tp'])
    fnr_orig = results_original['fn'] / (results_original['fn'] + results_original['tp'])
    
    print(f"\n3. False Negative Rate (Missing Mortality Cases):")
    print(f"   Model 1: {fnr_syn:.4f} ({fnr_syn*100:.2f}%)")
    print(f"   Model 2: {fnr_orig:.4f} ({fnr_orig*100:.2f}%)")
    
    if fnr_syn < fnr_orig:
        print("   → Model 1 misses fewer mortality cases (BETTER for clinical use)")
    else:
        print("   → Model 2 misses fewer mortality cases (BETTER for clinical use)")
    
    # False positive analysis
    fpr_syn = results_synthetic['fp'] / (results_synthetic['fp'] + results_synthetic['tn'])
    fpr_orig = results_original['fp'] / (results_original['fp'] + results_original['tn'])
    
    print(f"\n4. False Positive Rate (False Alarms):")
    print(f"   Model 1: {fpr_syn:.4f} ({fpr_syn*100:.2f}%)")
    print(f"   Model 2: {fpr_orig:.4f} ({fpr_orig*100:.2f}%)")
    
    # Precision analysis
    print(f"\n5. Precision Analysis (When predicting mortality, how often is it correct?):")
    print(f"   Model 1: {results_synthetic['precision']:.4f} ({results_synthetic['precision']*100:.2f}%)")
    print(f"   Model 2: {results_original['precision']:.4f} ({results_original['precision']*100:.2f}%)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BASELINE MODEL VISUALIZATION SCRIPT")
    print("="*80)
    print("\nThis script generates visualizations comparing the two baseline models.")
    print("Run 'evaluate_baselines.py' first to generate the evaluation results.")
    print("="*80)

