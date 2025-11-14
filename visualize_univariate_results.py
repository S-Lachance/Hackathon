"""
Create visualizations for univariate logistic regression results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_univariate_visualizations():
    """Create comprehensive visualizations for univariate logistic regression results."""

    print("Creating visualizations for univariate logistic regression results...")

    # Load univariate results
    df = pd.read_csv('univariate_logistic_results.csv')

    print(f"Loaded {len(df)} features from univariate analysis")
    print(f"Significant features (p < 0.1): {df['significant'].sum()}")
    print(".6f")
    print(".6f")

    # 1. P-value distribution histogram
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(df['p_value'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='p = 0.1 threshold')
    plt.axvline(x=0.05, color='orange', linestyle='--', linewidth=2, label='p = 0.05 threshold')
    plt.xlabel('p-value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of p-values\n(Univariate Logistic Regression)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # -log10 p-value for better visualization
    plt.subplot(1, 2, 2)
    log_p_values = -np.log10(df['p_value'] + 1e-300)  # Add small constant to avoid -inf
    plt.hist(log_p_values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(x=-np.log10(0.1), color='red', linestyle='--', linewidth=2, label='p = 0.1 (-log10 = 1.0)')
    plt.axvline(x=-np.log10(0.05), color='orange', linestyle='--', linewidth=2, label='p = 0.05 (-log10 = 1.3)')
    plt.xlabel('-log10(p-value)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('-log10 p-value Distribution\n(Higher values = more significant)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('univariate_pvalue_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: univariate_pvalue_distribution.png")

    # 2. Odds ratio distribution for significant features
    significant_df = df[df['significant'] == True].copy()
    significant_df['log_odds_ratio'] = np.log(significant_df['odds_ratio'])

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.hist(significant_df['odds_ratio'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(x=1.0, color='black', linestyle='-', linewidth=2, label='OR = 1.0 (no effect)')
    plt.xlabel('Odds Ratio', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Odds Ratio Distribution\n(Significant Features p < 0.1)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Volcano plot: effect size vs significance
    plt.subplot(2, 2, 2)
    colors = ['red' if or_val > 1 else 'blue' for or_val in significant_df['odds_ratio']]
    plt.scatter(significant_df['log_odds_ratio'], -np.log10(significant_df['p_value']),
               alpha=0.6, c=colors, s=30, edgecolors='black', linewidth=0.5)
    plt.axhline(y=-np.log10(0.1), color='red', linestyle='--', alpha=0.7, label='p = 0.1')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.7, label='OR = 1.0')
    plt.xlabel('log(Odds Ratio)', fontsize=12)
    plt.ylabel('-log10(p-value)', fontsize=12)
    plt.title('Volcano Plot: Effect Size vs Significance\n(Significant Features)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Top risk factors by odds ratio
    plt.subplot(2, 2, 3)
    top_risk = significant_df[significant_df['odds_ratio'] > 1].nlargest(15, 'odds_ratio')
    bars = plt.barh(range(len(top_risk)), top_risk['odds_ratio'], color='red', alpha=0.7)
    plt.yticks(range(len(top_risk)), top_risk['feature'])
    plt.xlabel('Odds Ratio', fontsize=12)
    plt.title('Top Risk Factors by Odds Ratio\n(Univariate Analysis)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add OR values on bars
    for i, (idx, row) in enumerate(top_risk.iterrows()):
        plt.text(row['odds_ratio'] + 0.1, i, '.1f', va='center', fontsize=8)

    # 5. Top protective factors by odds ratio (lowest ORs)
    plt.subplot(2, 2, 4)
    top_protective = significant_df[significant_df['odds_ratio'] < 1].nsmallest(15, 'odds_ratio')
    bars = plt.barh(range(len(top_protective)), top_protective['odds_ratio'], color='blue', alpha=0.7)
    plt.yticks(range(len(top_protective)), top_protective['feature'])
    plt.xlabel('Odds Ratio', fontsize=12)
    plt.title('Top Protective Factors by Odds Ratio\n(Univariate Analysis)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add OR values on bars
    for i, (idx, row) in enumerate(top_protective.iterrows()):
        plt.text(row['odds_ratio'] + 0.001, i, '.3f', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('univariate_odds_ratios_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: univariate_odds_ratios_analysis.png")

    # 6. Feature importance ranking by p-value
    plt.figure(figsize=(14, 10))

    # Sort by p-value (most significant first)
    sorted_df = df.nsmallest(30, 'p_value')

    # Create -log10 p-value for better visualization
    sorted_df['neg_log_p'] = -np.log10(sorted_df['p_value'] + 1e-300)

    bars = plt.barh(range(len(sorted_df)), sorted_df['neg_log_p'],
                   color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)

    plt.yticks(range(len(sorted_df)), sorted_df['feature'])
    plt.xlabel('-log10(p-value)', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title('Top 30 Features by Statistical Significance\n(Univariate Logistic Regression)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add significance threshold lines
    plt.axvline(x=-np.log10(0.1), color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='p = 0.1 threshold')
    plt.axvline(x=-np.log10(0.05), color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label='p = 0.05 threshold')

    plt.legend()
    plt.tight_layout()
    plt.savefig('univariate_significance_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: univariate_significance_ranking.png")

    # Print summary statistics
    print("\n" + "="*80)
    print("UNIVARIATE ANALYSIS SUMMARY")
    print("="*80)

    print("\n📊 Overall Statistics:")
    print(f"  • Total features analyzed: {len(df)}")
    print(f"  • Converged models: {df['converged'].sum()}")
    print(f"  • Failed to converge: {len(df) - df['converged'].sum()}")
    print(".6f")
    print(".6f")

    print("\n🎯 Significance Thresholds:")
    print(f"  • p < 0.1 (significant): {len(df[df['p_value'] < 0.1])} features")
    print(f"  • p < 0.05 (highly significant): {len(df[df['p_value'] < 0.05])} features")
    print(f"  • p < 0.01 (very significant): {len(df[df['p_value'] < 0.01])} features")

    print("\n🔴 Top 5 Risk Factors (by Odds Ratio):")
    top_risk = df[df['odds_ratio'] > 1].nlargest(5, 'odds_ratio')
    for i, (_, row) in enumerate(top_risk.iterrows(), 1):
        print("2d")

    print("\n🟢 Top 5 Protective Factors (by Odds Ratio):")
    top_protective = df[df['odds_ratio'] < 1].nsmallest(5, 'odds_ratio')
    for i, (_, row) in enumerate(top_protective.iterrows(), 1):
        print("2d")

    print("\n📁 Generated Files:")
    print("  • univariate_pvalue_distribution.png - p-value distributions")
    print("  • univariate_odds_ratios_analysis.png - odds ratios and volcano plot")
    print("  • univariate_significance_ranking.png - statistical significance ranking")

def create_univariate_report():
    """Create a comprehensive report of univariate analysis."""

    df = pd.read_csv('univariate_logistic_results.csv')

    with open('univariate_analysis_report.md', 'w') as f:
        f.write("# Univariate Logistic Regression Analysis Report\n\n")

        f.write("## Overview\n")
        f.write("This analysis performs univariate logistic regression on all features to assess their individual association with mortality (oym=TRUE).\n\n")
        f.write("**Methodology**: Each feature is tested individually against the outcome using logistic regression.\n\n")

        f.write("## Dataset\n")
        f.write(f"- Total features analyzed: {len(df)}\n")
        f.write(f"- Features with p < 0.1: {len(df[df['p_value'] < 0.1])}\n")
        f.write(f"- Models converged: {df['converged'].sum()}/{len(df)}\n\n")

        f.write("## Top 10 Risk Factors\n")
        f.write("| Rank | Feature | Odds Ratio | p-value | 95% CI |\n")
        f.write("|------|---------|------------|---------|--------|\n")
        risk_factors = df[df['odds_ratio'] > 1].nlargest(10, 'odds_ratio')
        for i, (_, row) in enumerate(risk_factors.iterrows(), 1):
            f.write("2d")

        f.write("\n## Top 10 Protective Factors\n")
        f.write("| Rank | Feature | Odds Ratio | p-value | 95% CI |\n")
        f.write("|------|---------|------------|---------|--------|\n")
        protective_factors = df[df['odds_ratio'] < 1].nsmallest(10, 'odds_ratio')
        for i, (_, row) in enumerate(protective_factors.iterrows(), 1):
            f.write("2d")

        f.write("\n## Statistical Summary\n")
        f.write(f"- Mean odds ratio: {df['odds_ratio'].mean():.3f}\n")
        f.write(f"- Median p-value: {df['p_value'].median():.2e}\n")
        f.write(f"- Features with OR > 2: {len(df[df['odds_ratio'] > 2])}\n")
        f.write(f"- Features with OR < 0.5: {len(df[df['odds_ratio'] < 0.5])}\n\n")

        f.write("## Interpretation\n")
        f.write("### Odds Ratios\n")
        f.write("- **OR > 1**: Risk factor (increases mortality probability)\n")
        f.write("- **OR < 1**: Protective factor (decreases mortality probability)\n")
        f.write("- **OR = 1**: No association\n\n")

        f.write("### p-values\n")
        f.write("- **p < 0.05**: Statistically significant\n")
        f.write("- **p < 0.01**: Highly significant\n")
        f.write("- **p < 0.001**: Very highly significant\n\n")

        f.write("### Feature Selection\n")
        f.write("Features with p < 0.1 were selected for multivariate analysis to balance statistical significance with the risk of missing potentially important variables.\n\n")

        f.write("## Files Generated\n")
        f.write("- `univariate_logistic_results.csv` - Complete univariate results\n")
        f.write("- `univariate_pvalue_distribution.png` - p-value distribution plots\n")
        f.write("- `univariate_odds_ratios_analysis.png` - Odds ratios and volcano plot\n")
        f.write("- `univariate_significance_ranking.png` - Statistical significance ranking\n")

    print("✓ Created: univariate_analysis_report.md")

if __name__ == "__main__":
    create_univariate_visualizations()
    create_univariate_report()
