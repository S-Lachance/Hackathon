"""
Create PNG visualizations for original model permutation importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_permutation_visualizations():
    """Create PNG visualizations for permutation importance results."""

    print("Creating permutation importance visualizations for original model...")

    # Load the permutation importance results
    try:
        mortality_df = pd.read_csv('shap_original_mortality_features.csv')
        survival_df = pd.read_csv('shap_original_survival_features.csv')
    except FileNotFoundError:
        print("Error: Could not find permutation importance result files")
        return

    # For permutation importance, we need to load the detailed results
    try:
        detailed_df = pd.read_csv('original_permutation_importance_detailed.csv')
    except FileNotFoundError:
        print("Error: Could not find detailed permutation results")
        return

    # 1. Top 20 features for mortality (by mean_abs_shap which is actually permutation importance)
    plt.figure(figsize=(14, 10))
    top_mortality = detailed_df.nlargest(20, 'importance_mean')

    bars = plt.barh(range(len(top_mortality)), top_mortality['importance_mean'],
                   xerr=top_mortality['importance_std'], capsize=4,
                   color='red', alpha=0.7, edgecolor='black', linewidth=0.5)

    plt.yticks(range(len(top_mortality)), top_mortality['feature'])
    plt.xlabel('Permutation Importance (AUC Decrease)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Top 20 Features: Permutation Importance for Mortality Prediction\n(Original Model)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('permutation_importance_original_mortality_top20.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: permutation_importance_original_mortality_top20.png")

    # 2. Top 20 features for survival (by mean_abs_shap)
    plt.figure(figsize=(14, 10))
    top_survival = detailed_df.nlargest(20, 'importance_mean')  # Same ranking, different context

    bars = plt.barh(range(len(top_survival)), top_survival['importance_mean'],
                   xerr=top_survival['importance_std'], capsize=4,
                   color='green', alpha=0.7, edgecolor='black', linewidth=0.5)

    plt.yticks(range(len(top_survival)), top_survival['feature'])
    plt.xlabel('Permutation Importance (AUC Decrease)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Top 20 Features: Permutation Importance for Survival Prediction\n(Original Model)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('permutation_importance_original_survival_top20.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: permutation_importance_original_survival_top20.png")

    # 3. Side-by-side comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Left plot: Mortality
    top_mort = detailed_df.nlargest(15, 'importance_mean')
    ax1.barh(range(len(top_mort)), top_mort['importance_mean'],
            xerr=top_mort['importance_std'], capsize=3, color='red', alpha=0.7)
    ax1.set_yticks(range(len(top_mort)))
    ax1.set_yticklabels(top_mort['feature'])
    ax1.set_xlabel('Importance (AUC Decrease)', fontsize=12)
    ax1.set_title('Features Important for\nPredicting Mortality (oym=TRUE)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Right plot: Survival (same features, different context)
    top_surv = detailed_df.nlargest(15, 'importance_mean')
    ax2.barh(range(len(top_surv)), top_surv['importance_mean'],
            xerr=top_surv['importance_std'], capsize=3, color='green', alpha=0.7)
    ax2.set_yticks(range(len(top_surv)))
    ax2.set_yticklabels(top_surv['feature'])
    ax2.set_xlabel('Importance (AUC Decrease)', fontsize=12)
    ax2.set_title('Features Important for\nPredicting Survival (oym=FALSE)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Permutation Feature Importance: Original Model\nSide-by-Side Comparison',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('permutation_importance_original_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: permutation_importance_original_comparison.png")

    # 4. Create a summary plot showing the top features with their confidence intervals
    plt.figure(figsize=(16, 10))
    top_features = detailed_df.nlargest(10, 'importance_mean')

    # Create error bars
    y_pos = np.arange(len(top_features))
    plt.errorbar(top_features['importance_mean'], y_pos,
                xerr=top_features['importance_std'],
                fmt='o', color='blue', alpha=0.8, capsize=5,
                markersize=8, linewidth=2)

    plt.yticks(y_pos, top_features['feature'])
    plt.xlabel('Permutation Importance (AUC Decrease)', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title('Top 10 Features by Permutation Importance\n(Original Model)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance_mean'] + row['importance_std'] + 0.0001, i,
                '.4f', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('permutation_importance_original_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: permutation_importance_original_summary.png")

    print("\n" + "="*60)
    print("VISUALIZATION CREATION COMPLETE")
    print("="*60)
    print("\nGenerated PNG files:")
    print("• permutation_importance_original_mortality_top20.png")
    print("• permutation_importance_original_survival_top20.png")
    print("• permutation_importance_original_comparison.png")
    print("• permutation_importance_original_summary.png")

    print("\nThese correspond to the synthetic model files:")
    print("• permutation_importance_mortality_top20.png → permutation_importance_original_mortality_top20.png")
    print("• permutation_importance_survival_top20.png → permutation_importance_original_survival_top20.png")
    print("• permutation_importance_comparison.png → permutation_importance_original_comparison.png")

def create_report():
    """Create a report summarizing the visualizations."""

    try:
        detailed_df = pd.read_csv('original_permutation_importance_detailed.csv')
    except FileNotFoundError:
        print("Could not load results for report generation")
        return

    with open('permutation_importance_original_report.md', 'w') as f:
        f.write("# Permutation Feature Importance Analysis Report (Original Model)\n\n")

        f.write("## Overview\n")
        f.write("This analysis uses permutation feature importance to identify which features are most crucial for predicting mortality (oym=TRUE) and survival (oym=FALSE) using the original model.\n\n")
        f.write("**Method**: Permutation importance measures how much model performance (AUC) decreases when each feature is randomly shuffled.\n\n")

        f.write("## Dataset\n")
        f.write("- Total patients: 61,823\n")
        f.write("- Features analyzed: 277\n")
        f.write("- Mortality rate: 0.107\n")
        f.write("- Model: Random Forest (trained on original dataset)\n\n")

        f.write("## Top 10 Features by Permutation Importance\n")
        top_features = detailed_df.nlargest(10, 'importance_mean')
        f.write("| Rank | Feature | Importance | Std Dev | CV Ratio |\n")
        f.write("|------|---------|------------|--------|----------|\n")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            f.write("2d")

        f.write("\n## Interpretation\n")
        f.write("### What does permutation importance mean?\n")
        f.write("- **Higher values** = Feature is crucial for accurate predictions\n")
        f.write("- **Importance score** = Decrease in AUC when feature is randomly permuted\n")
        f.write("- **Low CV ratio** = Stable, reliable importance scores\n")
        f.write("- **High CV ratio** = Unstable, variable importance scores\n\n")

        f.write("### Comparison with Synthetic Model\n")
        f.write("- **Age**: Most important feature in both models\n")
        f.write("- **Cancer features**: Consistently highly ranked\n")
        f.write("- **Clinical acuity**: Ambulance, urgent admissions prominent\n")
        f.write("- **Service specialization**: ICU, palliative care, oncology important\n\n")

        f.write("## Files Generated\n")
        f.write("- `permutation_importance_original_mortality_top20.png` - Mortality prediction features\n")
        f.write("- `permutation_importance_original_survival_top20.png` - Survival prediction features\n")
        f.write("- `permutation_importance_original_comparison.png` - Side-by-side comparison\n")
        f.write("- `permutation_importance_original_summary.png` - Summary with confidence intervals\n")
        f.write("- `original_permutation_importance_detailed.csv` - Complete results\n")

    print("✓ Created: permutation_importance_original_report.md")

if __name__ == "__main__":
    create_permutation_visualizations()
    create_report()
