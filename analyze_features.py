"""
Feature Analysis for Hospital Mortality Prediction
Implements the "Top Predictor" analysis strategy from the context primer
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List
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

def analyze_single_feature_impact(data, target_col='oym'):
    """
    Analysis 1: Single Feature Impact
    Shows mortality rate for key categorical features
    """
    print("\n" + "="*80)
    print("ANALYSIS 1: SINGLE FEATURE IMPACT")
    print("="*80)
    
    # Key features to analyze
    key_features = ['gender', 'living_status', 'admission_group', 'service_group', 'is_ambulance']
    
    results = []
    for feature in key_features:
        if feature in data.columns:
            grouped = data.groupby(feature)[target_col].agg(['mean', 'count'])
            grouped['feature'] = feature
            grouped['value'] = grouped.index
            grouped.columns = ['mortality_rate', 'patient_count', 'feature', 'value']
            results.append(grouped.reset_index(drop=True))
            
            print(f"\n{feature}:")
            for idx, row in grouped.iterrows():
                print(f"  {row['value']}: {row['mortality_rate']*100:.2f}% mortality ({row['patient_count']:,} patients)")
    
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def analyze_age_distribution(data, target_col='oym'):
    """
    Analysis 2: Age Analysis
    Bins age into 10-year groups and shows mortality curve
    """
    print("\n" + "="*80)
    print("ANALYSIS 2: AGE DISTRIBUTION AND MORTALITY")
    print("="*80)
    
    if 'age_original' not in data.columns:
        print("⚠ 'age_original' column not found in dataset")
        return None
    
    # Create age bins
    bins = list(range(0, 101, 10)) + [120]  # 0-10, 10-20, ..., 90-100, 100+
    labels = [f"{i}-{i+9}" for i in range(0, 100, 10)] + ["100+"]
    
    data_with_age = data.copy()
    data_with_age['age_group'] = pd.cut(data_with_age['age_original'], bins=bins, labels=labels, right=False)
    
    # Calculate mortality by age group
    age_analysis = data_with_age.groupby('age_group')[target_col].agg([
        ('mortality_rate', 'mean'),
        ('patient_count', 'count')
    ]).reset_index()
    
    print("\nMortality by Age Group:")
    print(age_analysis.to_string(index=False))
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x_pos = range(len(age_analysis))
    ax.bar(x_pos, age_analysis['mortality_rate'] * 100, color='steelblue', edgecolor='black')
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Mortality Rate (%)', fontsize=12)
    ax.set_title('One-Year Mortality Rate by Age Group', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(age_analysis['age_group'], rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Add patient counts as text
    for i, (rate, count) in enumerate(zip(age_analysis['mortality_rate'], age_analysis['patient_count'])):
        ax.text(i, rate * 100 + 1, f"n={count:,}", ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('age_mortality_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: age_mortality_analysis.png")
    
    return age_analysis

def analyze_top_predictors(data, target_col='oym', min_patient_count=100):
    """
    Analysis 3: The "Top Predictor" Report (MOST CRITICAL)
    Analyzes all diagnosis columns to find strongest predictors
    """
    print("\n" + "="*80)
    print("ANALYSIS 3: TOP PREDICTOR ANALYSIS (ALL DIAGNOSES)")
    print("="*80)
    print(f"Filtering: Only features with at least {min_patient_count} patients")
    
    # Find all diagnosis columns (dx_* and adm_*)
    dx_cols = [col for col in data.columns if col.startswith('dx_')]
    adm_cols = [col for col in data.columns if col.startswith('adm_')]
    all_diagnosis_cols = dx_cols + adm_cols
    
    print(f"\n✓ Found {len(all_diagnosis_cols)} diagnosis features:")
    print(f"  - {len(dx_cols)} comorbidity diagnoses (dx_*)")
    print(f"  - {len(adm_cols)} admission diagnoses (adm_*)")
    
    # Step 1: Unpivot (melt) the diagnosis columns
    print("\n⏳ Melting data into long format...")
    id_cols = ['patient_id', 'visit_id', target_col]
    melted = data[id_cols + all_diagnosis_cols].melt(
        id_vars=id_cols,
        value_vars=all_diagnosis_cols,
        var_name='diagnosis_feature',
        value_name='has_diagnosis'
    )
    
    # Step 2: Filter to only rows where patient HAS the diagnosis
    print("⏳ Filtering to positive diagnoses...")
    melted_positive = melted[melted['has_diagnosis'] == 1].copy()
    
    # Step 3: Group by diagnosis and calculate statistics
    print("⏳ Calculating mortality rates and patient counts...")
    predictor_analysis = melted_positive.groupby('diagnosis_feature')[target_col].agg([
        ('mortality_rate', 'mean'),
        ('patient_count', 'count')
    ]).reset_index()
    
    # Step 4: Filter by minimum patient count
    predictor_analysis = predictor_analysis[predictor_analysis['patient_count'] >= min_patient_count].copy()
    
    # Step 5: Sort by mortality rate (descending)
    predictor_analysis = predictor_analysis.sort_values('mortality_rate', ascending=False).reset_index(drop=True)
    
    # Add rank
    predictor_analysis.insert(0, 'rank', range(1, len(predictor_analysis) + 1))
    
    # Calculate overall baseline mortality for comparison
    baseline_mortality = data[target_col].mean()
    predictor_analysis['vs_baseline'] = predictor_analysis['mortality_rate'] - baseline_mortality
    predictor_analysis['relative_risk'] = predictor_analysis['mortality_rate'] / baseline_mortality
    
    print(f"\n✓ Analysis complete!")
    print(f"  - Baseline mortality rate: {baseline_mortality*100:.2f}%")
    print(f"  - Features meeting criteria: {len(predictor_analysis)}")
    
    # Display top 30
    print("\n" + "="*80)
    print("TOP 30 PREDICTORS OF ONE-YEAR MORTALITY")
    print("="*80)
    print(f"{'Rank':<6} {'Feature':<30} {'Mortality':<12} {'Patients':<12} {'vs Baseline':<12} {'Rel. Risk':<10}")
    print("-"*80)
    
    for idx, row in predictor_analysis.head(30).iterrows():
        print(f"{row['rank']:<6} {row['diagnosis_feature']:<30} {row['mortality_rate']*100:>10.2f}% {row['patient_count']:>10,} {row['vs_baseline']*100:>+10.2f}% {row['relative_risk']:>9.2f}x")
    
    # Save full results
    predictor_analysis.to_csv('top_predictors_full.csv', index=False)
    print(f"\n✓ Full results saved: top_predictors_full.csv")
    
    # Create visualization of top predictors
    plot_top_predictors(predictor_analysis.head(20), baseline_mortality)
    
    return predictor_analysis

def plot_top_predictors(top_predictors, baseline_mortality):
    """Visualize top predictors"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Top 20 Mortality Risk Predictors', fontsize=16, fontweight='bold')
    
    # Plot 1: Mortality Rate
    ax1 = axes[0]
    y_pos = range(len(top_predictors))
    colors = plt.cm.Reds(top_predictors['mortality_rate'] / top_predictors['mortality_rate'].max())
    
    ax1.barh(y_pos, top_predictors['mortality_rate'] * 100, color=colors, edgecolor='black')
    ax1.axvline(baseline_mortality * 100, color='blue', linestyle='--', linewidth=2, label=f'Baseline ({baseline_mortality*100:.1f}%)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_predictors['diagnosis_feature'], fontsize=9)
    ax1.set_xlabel('Mortality Rate (%)', fontsize=12)
    ax1.set_title('Mortality Rate by Feature', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Patient Count (bubble chart showing both mortality and volume)
    ax2 = axes[1]
    scatter = ax2.scatter(
        top_predictors['patient_count'],
        top_predictors['mortality_rate'] * 100,
        s=top_predictors['patient_count'] / 10,  # Scale for visibility
        c=top_predictors['mortality_rate'],
        cmap='Reds',
        alpha=0.6,
        edgecolors='black',
        linewidth=1
    )
    
    # Annotate top 5
    for idx, row in top_predictors.head(5).iterrows():
        ax2.annotate(
            row['diagnosis_feature'],
            (row['patient_count'], row['mortality_rate'] * 100),
            fontsize=8,
            ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
        )
    
    ax2.axhline(baseline_mortality * 100, color='blue', linestyle='--', linewidth=2, label=f'Baseline ({baseline_mortality*100:.1f}%)')
    ax2.set_xlabel('Number of Patients', fontsize=12)
    ax2.set_ylabel('Mortality Rate (%)', fontsize=12)
    ax2.set_title('Risk vs Volume Trade-off', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.colorbar(scatter, ax=ax2, label='Mortality Rate')
    plt.tight_layout()
    plt.savefig('top_predictors_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: top_predictors_visualization.png")

def analyze_visit_patterns(data, target_col='oym'):
    """Analyze patterns in ED visits and other visit characteristics"""
    print("\n" + "="*80)
    print("ANALYSIS 4: VISIT PATTERNS")
    print("="*80)
    
    if 'ed_visit_count' in data.columns:
        # Bin ED visits
        ed_bins = [0, 1, 2, 3, 5, 10, 1000]
        ed_labels = ['0', '1', '2', '3-4', '5-9', '10+']
        data_copy = data.copy()
        data_copy['ed_visit_group'] = pd.cut(data_copy['ed_visit_count'], bins=ed_bins, labels=ed_labels, right=False)
        
        ed_analysis = data_copy.groupby('ed_visit_group')[target_col].agg([
            ('mortality_rate', 'mean'),
            ('patient_count', 'count')
        ]).reset_index()
        
        print("\nMortality by ED Visit Count:")
        print(ed_analysis.to_string(index=False))
    
    return ed_analysis if 'ed_visit_count' in data.columns else None

def main():
    """Main analysis pipeline"""
    
    print("\n" + "="*80)
    print("FEATURE ANALYSIS FOR MORTALITY PREDICTION")
    print("RSN Hackathon Challenge: Hospital One-year Mortality Risk Prediction")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    data = pd.read_csv('csv/dataset.csv')
    print(f"✓ Loaded: {data.shape[0]:,} rows, {data.shape[1]} columns")
    
    target_col = 'oym'
    if target_col not in data.columns:
        print(f"ERROR: Target column '{target_col}' not found!")
        return
    
    print(f"✓ Target: {data[target_col].sum():,} deaths ({data[target_col].mean()*100:.2f}% mortality rate)")
    
    # Run all analyses
    single_feature_results = analyze_single_feature_impact(data, target_col)
    age_results = analyze_age_distribution(data, target_col)
    top_predictors = analyze_top_predictors(data, target_col, min_patient_count=100)
    visit_patterns = analyze_visit_patterns(data, target_col)
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - KEY TAKEAWAYS")
    print("="*80)
    print("\n📊 Generated Files:")
    print("  ✓ top_predictors_full.csv - Complete ranking of all diagnosis features")
    print("  ✓ top_predictors_visualization.png - Visual analysis of top 20 features")
    print("  ✓ age_mortality_analysis.png - Age distribution analysis")
    print("\n💡 Next Steps:")
    print("  1. Review top_predictors_full.csv to select features for your model")
    print("  2. Consider feature engineering based on high-risk diagnoses")
    print("  3. Use these insights to inform your XGBoost/LightGBM model")
    print("  4. Test different feature subsets (top 10, top 50, top 100, etc.)")
    print("\n")

if __name__ == "__main__":
    main()

