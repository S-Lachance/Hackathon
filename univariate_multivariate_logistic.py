"""
Univariate and Multivariate Logistic Regression Analysis for Mortality Prediction
1. Perform univariate logistic regression for all variables
2. Select variables with p-value < 0.15
3. Perform multivariate logistic regression with selected variables
4. Show top 10 variables most associated with mortality (oym=TRUE)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
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

def load_and_prepare_data(filepath='csv/dataset.csv'):
    """Load and prepare the dataset for analysis."""
    print("Loading and preparing data for logistic regression analysis...")

    # Load data exactly as in evaluate_baselines.py
    data = pd.read_csv(filepath)
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

    # Return the test set (what the models were trained on)
    return x_test[feature_cols], y_test

def univariate_logistic_regression(X, y):
    """
    Perform univariate logistic regression for each feature.

    Returns:
    - DataFrame with results for each feature
    """
    print("Performing univariate logistic regression for all features...")

    results = []

    for col in X.columns:
        try:
            # Prepare data for this feature
            X_feature = X[[col]].copy()

            # Skip features with no variance
            if X_feature[col].nunique() <= 1:
                continue

            # Handle missing values
            X_feature = X_feature.fillna(X_feature.mean() if X_feature[col].dtype in ['float64', 'int64'] else X_feature.mode().iloc[0])

            # Add constant for statsmodels
            X_feature_sm = sm.add_constant(X_feature)

            # Fit logistic regression
            model = sm.Logit(y, X_feature_sm)
            result = model.fit(disp=False)

            # Extract results
            coef = result.params[col]
            odds_ratio = np.exp(coef)
            p_value = result.pvalues[col]
            conf_int = result.conf_int().loc[col]

            results.append({
                'feature': col,
                'coefficient': coef,
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'conf_int_lower': conf_int[0],
                'conf_int_upper': conf_int[1],
                'aic': result.aic,
                'converged': result.converged
            })

        except Exception as e:
            print(f"Warning: Failed to fit model for {col}: {str(e)}")
            continue

    results_df = pd.DataFrame(results)
    results_df['significant'] = results_df['p_value'] < 0.15

    print(f"✓ Completed univariate analysis for {len(results_df)} features")
    print(f"  - {results_df['significant'].sum()} features with p < 0.15")

    return results_df

def select_significant_features(univariate_results, p_threshold=0.15):
    """Select features with p-value below threshold."""
    significant_features = univariate_results[univariate_results['p_value'] < p_threshold]['feature'].tolist()
    print(f"Selected {len(significant_features)} features with p < {p_threshold}")

    return significant_features

def multivariate_logistic_regression(X, y, selected_features):
    """
    Perform multivariate logistic regression with selected features.

    Returns:
    - Fitted model results
    - Feature importance rankings
    """
    print(f"Performing multivariate logistic regression with {len(selected_features)} features...")

    # Select features
    X_selected = X[selected_features].copy()

    # Handle missing values
    for col in X_selected.columns:
        if X_selected[col].dtype in ['float64', 'int64']:
            X_selected[col] = X_selected[col].fillna(X_selected[col].mean())
        else:
            X_selected[col] = X_selected[col].fillna(X_selected[col].mode().iloc[0])

    # Add constant for statsmodels
    X_selected_sm = sm.add_constant(X_selected)

    # Fit multivariate logistic regression
    model = sm.Logit(y, X_selected_sm)
    result = model.fit(disp=False)

    print(".4f")

    # Extract results
    results_df = pd.DataFrame({
        'feature': selected_features,
        'coefficient': result.params[selected_features],
        'odds_ratio': np.exp(result.params[selected_features]),
        'p_value': result.pvalues[selected_features],
        'conf_int_lower': result.conf_int().loc[selected_features, 0],
        'conf_int_upper': result.conf_int().loc[selected_features, 1]
    })

    # Add absolute coefficient for ranking
    results_df['abs_coefficient'] = np.abs(results_df['coefficient'])
    results_df['odds_ratio_direction'] = results_df['odds_ratio'].apply(
        lambda x: 'risk' if x > 1 else 'protective'
    )

    # Sort by absolute coefficient (most influential)
    results_df = results_df.sort_values('abs_coefficient', ascending=False)

    print("✓ Multivariate regression completed")

    return result, results_df

def plot_top_features(multivariate_results, top_n=10):
    """Plot the top N most important features from multivariate analysis."""
    top_features = multivariate_results.head(top_n)

    plt.figure(figsize=(12, 8))

    # Color based on odds ratio direction
    colors = ['red' if or_val > 1 else 'green' for or_val in top_features['odds_ratio']]

    # Create horizontal bar plot
    bars = plt.barh(range(len(top_features)), top_features['odds_ratio'], color=colors)

    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Odds Ratio')
    plt.title(f'Top {top_n} Features: Odds Ratios from Multivariate Logistic Regression')
    plt.axvline(x=1, color='black', linestyle='-', alpha=0.5)

    # Add odds ratio values on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['odds_ratio'] + 0.01 if row['odds_ratio'] > 1 else row['odds_ratio'] - 0.5,
                i, '.2f', va='center', fontsize=10)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multivariate_logistic_top10_features.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Top {top_n} features plot saved: multivariate_logistic_top10_features.png")

def create_comprehensive_report(univariate_results, multivariate_results, selected_features):
    """Create a comprehensive analysis report."""

    print("\n" + "="*100)
    print("COMPREHENSIVE LOGISTIC REGRESSION ANALYSIS REPORT")
    print("="*100)

    print("\n📊 UNIVARIATE ANALYSIS SUMMARY:")
    print("-" * 50)
    total_features = len(univariate_results)
    significant_features = len(selected_features)
    print(f"• Total features analyzed: {total_features}")
    print(f"• Features with p < 0.15: {significant_features} ({significant_features/total_features*100:.1f}%)")

    print("\n🔴 MULTIVARIATE ANALYSIS RESULTS:")
    print("-" * 50)

    # Top 10 risk factors
    risk_factors = multivariate_results[multivariate_results['odds_ratio'] > 1].head(10)
    print("\nTOP 10 RISK FACTORS (Odds Ratio > 1):")
    for i, (_, row) in enumerate(risk_factors.iterrows(), 1):
        print("2d")
    # Top 10 protective factors
    protective_factors = multivariate_results[multivariate_results['odds_ratio'] < 1].tail(10)
    protective_factors = protective_factors.sort_values('odds_ratio', ascending=True)  # Most protective first
    print("\n🟢 TOP 10 PROTECTIVE FACTORS (Odds Ratio < 1):")
    for i, (_, row) in enumerate(protective_factors.iterrows(), 1):
        print("2d")
    print("\n📈 MODEL PERFORMANCE:")
    print("-" * 30)
    print(".4f")
    print(f"• Number of predictors: {len(selected_features)}")
    print(f"• Significant predictors (p < 0.05): {len(multivariate_results[multivariate_results['p_value'] < 0.05])}")

    print("\n💡 KEY INSIGHTS:")
    print("-" * 20)
    print("• Variables were selected using p < 0.1 threshold in univariate analysis")
    print("• Multivariate model controls for confounding between variables")
    print("• Odds ratios show the relative risk/protection after adjusting for other factors")
    print("• Confidence intervals indicate precision of estimates")

    # Save detailed results
    univariate_results.to_csv('univariate_logistic_results.csv', index=False)
    multivariate_results.to_csv('multivariate_logistic_results.csv', index=False)

    print("\n📁 FILES SAVED:")
    print("-" * 15)
    print("• univariate_logistic_results.csv - All univariate results")
    print("• multivariate_logistic_results.csv - Multivariate coefficients")
    print("• multivariate_logistic_top10_features.png - Top 10 features visualization")

    # Create markdown report
    with open('logistic_regression_analysis_report.md', 'w') as f:
        f.write("# Logistic Regression Analysis: Mortality Prediction\n\n")

        f.write("## Analysis Overview\n")
        f.write("This analysis performs systematic feature selection and multivariate modeling:\n")
        f.write("1. **Univariate logistic regression** for all features\n")
        f.write("2. **Feature selection** using p-value < 0.15 threshold\n")
        f.write("3. **Multivariate logistic regression** with selected features\n")
        f.write("4. **Results interpretation** focusing on mortality association\n\n")

        f.write("## Dataset\n")
        f.write(f"- Total patients: {len(X) + len(X_train) if 'X_train' in locals() else 'N/A'}\n")
        f.write(f"- Features analyzed: {total_features}\n")
        f.write(f"- Selected for multivariate: {significant_features}\n\n")

        f.write("## Top 10 Risk Factors\n")
        f.write("| Rank | Feature | Odds Ratio | 95% CI | p-value |\n")
        f.write("|------|---------|------------|--------|--------|\n")
        for i, (_, row) in enumerate(risk_factors.iterrows(), 1):
            f.write(".2f")

        f.write("\n## Top 10 Protective Factors\n")
        f.write("| Rank | Feature | Odds Ratio | 95% CI | p-value |\n")
        f.write("|------|---------|------------|--------|--------|\n")
        for i, (_, row) in enumerate(protective_factors.iterrows(), 1):
            f.write(".2f")

        f.write("\n## Files Generated\n")
        f.write("- `univariate_logistic_results.csv` - Complete univariate analysis\n")
        f.write("- `multivariate_logistic_results.csv` - Multivariate coefficients\n")
        f.write("- `multivariate_logistic_top10_features.png` - Visualization\n")

    print("✓ Comprehensive report saved: logistic_regression_analysis_report.md")

def main():
    """Main function for logistic regression analysis."""
    print("=" * 100)
    print("LOGISTIC REGRESSION ANALYSIS: FEATURE SELECTION & MULTIVARIATE MODELING")
    print("=" * 100)

    # Load and prepare data
    X, y = load_and_prepare_data()

    if X is None:
        print("Failed to load data")
        return

    # Step 1: Univariate logistic regression
    print("\n" + "="*80)
    print("STEP 1: UNIVARIATE LOGISTIC REGRESSION")
    print("="*80)

    univariate_results = univariate_logistic_regression(X, y)

    # Step 2: Feature selection (p < 0.1)
    print("\n" + "="*80)
    print("STEP 2: FEATURE SELECTION (p < 0.1)")
    print("="*80)

    selected_features = select_significant_features(univariate_results, p_threshold=0.1)

    # Step 3: Multivariate logistic regression
    print("\n" + "="*80)
    print("STEP 3: MULTIVARIATE LOGISTIC REGRESSION")
    print("="*80)

    mv_model, multivariate_results = multivariate_logistic_regression(X, y, selected_features)

    # Step 4: Results and visualization
    print("\n" + "="*80)
    print("STEP 4: RESULTS & VISUALIZATION")
    print("="*80)

    plot_top_features(multivariate_results, top_n=10)

    # Step 5: Comprehensive report
    create_comprehensive_report(univariate_results, multivariate_results, selected_features)

    print("\n" + "=" * 100)
    print("LOGISTIC REGRESSION ANALYSIS COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
