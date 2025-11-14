"""
Create summary plot with confidence intervals for synthetic model permutation importance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_synthetic_summary_plot():
    """Create summary plot with confidence intervals for synthetic model."""

    print("Creating summary plot with confidence intervals for synthetic model...")

    # Load synthetic model permutation results
    synth_df = pd.read_csv('permutation_importance_mortality.csv')

    # Get top 10 features by absolute importance
    top_features = synth_df.nlargest(10, 'abs_importance')

    # Create the plot
    plt.figure(figsize=(16, 10))

    # Create error bars
    y_pos = np.arange(len(top_features))
    plt.errorbar(top_features['importance_mean'], y_pos,
                xerr=top_features['importance_std'],
                fmt='o', color='blue', alpha=0.8, capsize=5,
                markersize=8, linewidth=2, markerfacecolor='lightblue',
                markeredgecolor='darkblue', markeredgewidth=1)

    plt.yticks(y_pos, top_features['feature'])
    plt.xlabel('Permutation Importance (AUC Decrease)', fontsize=14, fontweight='bold')
    plt.ylabel('Features', fontsize=14, fontweight='bold')
    plt.title('Top 10 Features by Permutation Importance\n(Synthetic Model)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels with importance scores
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance_mean'] + row['importance_std'] + 0.0005, i,
                '.4f', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('permutation_importance_synthetic_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Created: permutation_importance_synthetic_summary.png")

    # Print summary statistics
    print("\nTop 10 Features Summary (Synthetic Model):")
    print("=" * 60)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print("2d")

    print("\nSummary Statistics:")
    print(".6f")
    print(".6f")
    print(".6f")

if __name__ == "__main__":
    create_synthetic_summary_plot()
