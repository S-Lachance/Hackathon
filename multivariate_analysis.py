"""
Multivariate Analysis for Hospital Mortality Prediction
Analyzes which features are most predictive of mortality (oym = TRUE)
using various statistical and machine learning approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath='csv/dataset.csv'):
    """Load and prepare the dataset for multivariate analysis."""
    print("Loading and preparing data for multivariate analysis...")

    # Load data using existing function
    from csv_handling import load_and_prep_data
    df = load_and_prep_data(filepath)

    if df is None:
        return None

    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Select one visit per patient (same as ML models)
    df = df.groupby("patient_id", group_keys=False).apply(
        lambda x: x.sample(1, random_state=42)
    ).reset_index(drop=True)
    print(f"Selected one visit per patient: {len(df)} rows")

    return df

def get_feature_groups(df):
    """Get different feature groups for analysis."""
    # Get column types from the homr.py logic
    dx_cols = [dx for dx in df.columns if dx.startswith('dx_')]
    adm_cols = [adm for adm in df.columns if adm.startswith('adm_')]
    cat_cols = ['gender', 'living_status', 'admission_group', 'service_group']
    cont_cols = ['age_original', 'ed_visit_count', 'ho_ambulance_count', 'total_duration']
    bin_cols = ['flu_season', 'is_ambulance', 'is_icu_start_ho', 'is_urg_readm', 'has_dx']

    return {
        'continuous': cont_cols,
        'binary': bin_cols + dx_cols + adm_cols,
        'categorical': cat_cols,
        'diagnosis': dx_cols,
        'admission': adm_cols
    }

def univariate_analysis(df, target_col='oym'):
    """Perform univariate statistical tests for each feature vs mortality."""
    print("\n" + "="*80)
    print("UNIVARIATE ANALYSIS")
    print("="*80)

    feature_groups = get_feature_groups(df)
    results = []

    # Analyze continuous features
    print("\n--- Continuous Features ---")
    for col in feature_groups['continuous']:
        if col in df.columns:
            survived = df[df[target_col] == 0][col].dropna()
            died = df[df[target_col] == 1][col].dropna()

            # T-test
            if len(survived) > 1 and len(died) > 1:
                t_stat, p_val = stats.ttest_ind(survived, died, equal_var=False)
                cohens_d = (died.mean() - survived.mean()) / np.sqrt((died.std()**2 + survived.std()**2) / 2)

                print(".3f")
                results.append({
                    'feature': col,
                    'type': 'continuous',
                    'test': 't-test',
                    'statistic': t_stat,
                    'p_value': p_val,
                    'effect_size': cohens_d,
                    'died_mean': died.mean(),
                    'survived_mean': survived.mean()
                })

    # Analyze binary features
    print("\n--- Binary Features (Chi-square test) ---")
    binary_features = [col for col in feature_groups['binary'] if col in df.columns and col != target_col]

    for col in binary_features[:20]:  # Limit to first 20 to avoid too much output
        if df[col].nunique() == 2:
            contingency_table = pd.crosstab(df[col], df[target_col])
            if contingency_table.shape == (2, 2):
                chi2, p_val, dof, expected = chi2_contingency(contingency_table)

                # Cramer's V effect size
                n = contingency_table.sum().sum()
                min_dim = min(contingency_table.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim))

                mortality_rate_with_feature = df[df[col] == 1][target_col].mean()
                mortality_rate_without_feature = df[df[col] == 0][target_col].mean()

                print(".3f")
                results.append({
                    'feature': col,
                    'type': 'binary',
                    'test': 'chi-square',
                    'statistic': chi2,
                    'p_value': p_val,
                    'effect_size': cramers_v,
                    'mortality_with_feature': mortality_rate_with_feature,
                    'mortality_without_feature': mortality_rate_without_feature
                })

    return pd.DataFrame(results)

def correlation_analysis(df, target_col='oym'):
    """Analyze correlations between features and mortality."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    feature_groups = get_feature_groups(df)

    # Prepare data for correlation
    df_corr = df.copy()

    # One-hot encode categorical variables
    cat_cols = [col for col in feature_groups['categorical'] if col in df.columns]
    if cat_cols:
        df_corr = pd.get_dummies(df_corr, columns=cat_cols, dtype=int, drop_first=True)

    # Select features (excluding target and patient_id)
    exclude_cols = [target_col, 'patient_id', 'CSO']
    feature_cols = [col for col in df_corr.columns if col not in exclude_cols and
                   df_corr[col].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"Analyzing correlations for {len(feature_cols)} features...")

    # Calculate point-biserial correlations for binary features
    correlations = []
    for col in feature_cols:
        if df_corr[col].nunique() == 2 and col != target_col:
            # Point-biserial correlation for binary features
            r_pb = stats.pointbiserialr(df_corr[col], df_corr[target_col])[0]
            correlations.append({'feature': col, 'correlation': r_pb, 'method': 'point-biserial'})
        else:
            # Pearson correlation for continuous features
            if df_corr[col].std() > 0:  # Avoid constant features
                r_pearson = df_corr[col].corr(df_corr[target_col])
                if not np.isnan(r_pearson):
                    correlations.append({'feature': col, 'correlation': r_pearson, 'method': 'pearson'})

    corr_df = pd.DataFrame(correlations)
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)

    print("\nTop 20 features by absolute correlation with mortality:")
    print(corr_df.head(20).to_string(index=False))

    return corr_df

def multivariate_logistic_regression(df, target_col='oym'):
    """Perform multivariate logistic regression analysis."""
    print("\n" + "="*80)
    print("MULTIVARIATE LOGISTIC REGRESSION")
    print("="*80)

    feature_groups = get_feature_groups(df)

    # Prepare features
    df_model = df.copy()

    # One-hot encode categorical variables
    cat_cols = [col for col in feature_groups['categorical'] if col in df.columns]
    if cat_cols:
        df_model = pd.get_dummies(df_model, columns=cat_cols, dtype=int, drop_first=True)

    # Select features
    exclude_cols = [target_col, 'patient_id', 'CSO']
    feature_cols = [col for col in df_model.columns if col not in exclude_cols and
                   df_model[col].dtype in ['int64', 'float64', 'int32', 'float32']]

    # Limit to top correlated features to avoid overfitting
    corr_df = correlation_analysis(df_model, target_col)
    top_features = corr_df.head(30)['feature'].tolist()
    feature_cols = [col for col in feature_cols if col in top_features]

    print(f"Using {len(feature_cols)} features in logistic regression...")

    # Prepare data
    X = df_model[feature_cols].fillna(0)  # Fill missing values with 0
    y = df_model[target_col]

    # Scale continuous features
    continuous_features = [col for col in feature_cols if col in feature_groups['continuous']]
    if continuous_features:
        scaler = StandardScaler()
        X[continuous_features] = scaler.fit_transform(X[continuous_features])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Fit logistic regression with regularization to handle multicollinearity
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
    lr_model.fit(X_train, y_train)

    # Get predictions
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(".4f")

    # Get coefficients
    coefficients = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': lr_model.coef_[0],
        'odds_ratio': np.exp(lr_model.coef_[0])
    }).sort_values('coefficient', ascending=False)

    print("\nTop 15 positive predictors (increase mortality risk):")
    print(coefficients.head(15)[['feature', 'odds_ratio']].to_string(index=False))

    print("\nTop 15 negative predictors (decrease mortality risk):")
    print(coefficients.tail(15)[['feature', 'odds_ratio']].to_string(index=False))

    return coefficients, auc_score

def feature_importance_analysis(df, target_col='oym'):
    """Analyze feature importance using multiple methods."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    feature_groups = get_feature_groups(df)

    # Prepare data
    df_fi = df.copy()

    # One-hot encode categorical variables
    cat_cols = [col for col in feature_groups['categorical'] if col in df.columns]
    if cat_cols:
        df_fi = pd.get_dummies(df_fi, columns=cat_cols, dtype=int, drop_first=True)

    # Select features
    exclude_cols = [target_col, 'patient_id', 'CSO']
    feature_cols = [col for col in df_fi.columns if col not in exclude_cols and
                   df_fi[col].dtype in ['int64', 'float64', 'int32', 'float32']]

    X = df_fi[feature_cols].fillna(0)
    y = df_fi[target_col]

    # Random Forest Feature Importance
    print("Computing Random Forest feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)

    # Mutual Information
    print("Computing Mutual Information scores...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': feature_cols,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)

    # ANOVA F-test (for continuous features)
    print("Computing ANOVA F-test scores...")
    continuous_mask = X.nunique() > 2
    if continuous_mask.sum() > 0:
        f_scores, f_pvals = f_classif(X.loc[:, continuous_mask], y)
        ftest_importance = pd.DataFrame({
            'feature': X.columns[continuous_mask],
            'f_score': f_scores,
            'f_pval': f_pvals
        }).sort_values('f_score', ascending=False)
    else:
        ftest_importance = pd.DataFrame(columns=['feature', 'f_score', 'f_pval'])

    # Combine results
    combined = pd.merge(rf_importance, mi_importance, on='feature', how='outer')
    combined = pd.merge(combined, ftest_importance, on='feature', how='outer')
    combined = combined.fillna(0)

    # Create composite score
    combined['composite_score'] = (
        combined['rf_importance'].rank(pct=True) +
        combined['mutual_info'].rank(pct=True) +
        combined['f_score'].rank(pct=True)
    ) / 3

    combined = combined.sort_values('composite_score', ascending=False)

    print("\nTop 20 features by composite importance:")
    top_features = combined.head(20)[['feature', 'composite_score', 'rf_importance', 'mutual_info', 'f_score']]
    print(top_features.to_string(index=False))

    return combined

def dimensionality_reduction_analysis(df, target_col='oym'):
    """Perform PCA and other dimensionality reduction analyses."""
    print("\n" + "="*80)
    print("DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*80)

    feature_groups = get_feature_groups(df)

    # Prepare data
    df_pca = df.copy()

    # One-hot encode categorical variables
    cat_cols = [col for col in feature_groups['categorical'] if col in df.columns]
    if cat_cols:
        df_pca = pd.get_dummies(df_pca, columns=cat_cols, dtype=int, drop_first=True)

    # Select features
    exclude_cols = [target_col, 'patient_id', 'CSO']
    feature_cols = [col for col in df_pca.columns if col not in exclude_cols and
                   df_pca[col].dtype in ['int64', 'float64', 'int32', 'float32']]

    X = df_pca[feature_cols].fillna(0)
    y = df_pca[target_col]

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(20, len(feature_cols)))
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(".2f")
    print(".2f")

    # Plot explained variance
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(explained_var) + 1), explained_var, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Variance')
    plt.grid(True, alpha=0.3)

    # PCA scatter plot colored by mortality
    plt.subplot(1, 3, 3)
    colors = ['blue' if outcome == 0 else 'red' for outcome in y]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=30)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA: Mortality Classes')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multivariate_pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ PCA analysis plot saved: multivariate_pca_analysis.png")

    return pca, explained_var, cumulative_var

def create_multivariate_report(df):
    """Create a comprehensive multivariate analysis report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MULTIVARIATE ANALYSIS REPORT")
    print("="*80)

    target_col = 'oym'

    # Basic dataset statistics
    mortality_rate = df[target_col].mean()
    print(".3f")
    print(f"Total patients: {len(df)}")
    print(f"Mortality cases: {df[target_col].sum()}")
    print(f"Survival cases: {(1 - df[target_col]).sum()}")

    # Run all analyses
    print("\n1. Running univariate analysis...")
    univariate_results = univariate_analysis(df, target_col)

    print("\n2. Running correlation analysis...")
    correlation_results = correlation_analysis(df, target_col)

    print("\n3. Running multivariate logistic regression...")
    try:
        lr_results, auc_score = multivariate_logistic_regression(df, target_col)
    except Exception as e:
        print(f"Logistic regression failed: {e}")
        lr_results, auc_score = None, None

    print("\n4. Running feature importance analysis...")
    try:
        importance_results = feature_importance_analysis(df, target_col)
    except Exception as e:
        print(f"Feature importance analysis failed: {e}")
        importance_results = None

    print("\n5. Running dimensionality reduction analysis...")
    try:
        pca_model, explained_var, cumulative_var = dimensionality_reduction_analysis(df, target_col)
    except Exception as e:
        print(f"Dimensionality reduction failed: {e}")
        pca_model, explained_var, cumulative_var = None, None, None

    # Save results
    print("\nSaving results...")

    if univariate_results is not None:
        univariate_results.to_csv('multivariate_univariate_results.csv', index=False)
        print("✓ Univariate results saved: multivariate_univariate_results.csv")

    if correlation_results is not None:
        correlation_results.to_csv('multivariate_correlations.csv', index=False)
        print("✓ Correlation results saved: multivariate_correlations.csv")

    if lr_results is not None:
        lr_results.to_csv('multivariate_logistic_coefficients.csv', index=False)
        print("✓ Logistic regression coefficients saved: multivariate_logistic_coefficients.csv")

    if importance_results is not None:
        importance_results.to_csv('multivariate_feature_importance.csv', index=False)
        print("✓ Feature importance results saved: multivariate_feature_importance.csv")

    # Create summary report
    with open('multivariate_analysis_report.md', 'w') as f:
        f.write("# Multivariate Analysis Report: Hospital Mortality Prediction\n\n")

        f.write("## Dataset Overview\n")
        f.write(f"- Total patients: {len(df)}\n")
        f.write(f"- Mortality rate: {mortality_rate:.3f} ({df[target_col].sum()} deaths)\n\n")

        if auc_score is not None:
            f.write(f"## Model Performance\n")
            f.write(f"- Logistic Regression AUC: {auc_score:.4f}\n\n")

        if correlation_results is not None and len(correlation_results) > 0:
            f.write("## Top Correlated Features\n")
            f.write("| Feature | Correlation | Method |\n")
            f.write("|---------|-------------|--------|\n")
            for _, row in correlation_results.head(10).iterrows():
                f.write(f"| {row['feature']} | {row['correlation']:.3f} | {row['method']} |\n")
            f.write("\n")

        if lr_results is not None and len(lr_results) > 0:
            f.write("## Key Predictors from Logistic Regression\n")
            f.write("### Risk Factors (Odds Ratio > 1)\n")
            risk_factors = lr_results[lr_results['odds_ratio'] > 1].head(10)
            f.write("| Feature | Odds Ratio |\n")
            f.write("|---------|------------|\n")
            for _, row in risk_factors.iterrows():
                f.write(f"| {row['feature']} | {row['odds_ratio']:.2f} |\n")

            f.write("\n### Protective Factors (Odds Ratio < 1)\n")
            protective = lr_results[lr_results['odds_ratio'] < 1].tail(10)
            f.write("| Feature | Odds Ratio |\n")
            f.write("|---------|------------|\n")
            for _, row in protective.iterrows():
                f.write(f"| {row['feature']} | {row['odds_ratio']:.2f} |\n")
            f.write("\n")

        f.write("## Analysis Files Generated\n")
        f.write("- `multivariate_univariate_results.csv` - Univariate statistical tests\n")
        f.write("- `multivariate_correlations.csv` - Feature correlations with mortality\n")
        f.write("- `multivariate_logistic_coefficients.csv` - Logistic regression coefficients\n")
        f.write("- `multivariate_feature_importance.csv` - Multi-method feature importance\n")
        f.write("- `multivariate_pca_analysis.png` - PCA dimensionality reduction plots\n")

    print("✓ Comprehensive report saved: multivariate_analysis_report.md")

    print("\n" + "="*80)
    print("MULTIVARIATE ANALYSIS COMPLETE")
    print("="*80)

def main():
    """Main function to run all multivariate analyses."""
    # Load data
    df = load_and_prepare_data()

    if df is None:
        print("Failed to load data")
        return

    # Run comprehensive multivariate analysis
    create_multivariate_report(df)

if __name__ == "__main__":
    main()
