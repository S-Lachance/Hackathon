import pandas as pd
import numpy as np

# --- Configuration ---

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! UPDATE THIS with the path to your dataset file
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FILEPATH = 'csv/dataset.csv'  # Or 'your_data.xlsx'

TARGET_COLUMN = 'oym'

# --- End of Configuration ---


def load_and_prep_data(filepath):
    """
    Loads the data and prepares it for analysis.
    - Loads from CSV or Excel (auto-detects).
    - Converts all logical/boolean columns to integers (0/1).
    """
    print(f"Loading data from {filepath}...")
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            print(f"Error: Unknown file type for {filepath}")
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please update the FILEPATH variable in this script.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    print("Data loaded. Converting logicals to integers (0/1)...")
    
    # Convert all boolean columns (e.g., TRUE/FALSE) to 0/1
    for col in df.select_dtypes(include=['bool']).columns:
        print(f"Converting bool column: {col}")
        df[col] = df[col].astype(int)
        
    # The user's list also had 'logical' types for 'oym' and 'CSO'
    # which pandas might read as 'object' (strings 'TRUE'/'FALSE').
    # This handles that case.
    for col in [TARGET_COLUMN, 'CSO']:
        if col in df.columns and df[col].dtype == 'object':
            print(f"Converting object column: {col}")
            # Map 'TRUE' (or similar) to 1, and everything else to 0
            # Adjust this if your logicals are 'T'/'F' or 'Yes'/'No'
            if isinstance(df[col].iloc[0], str):
                # Assuming 'TRUE'/'FALSE' strings
                df[col] = df[col].str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)
            elif isinstance(df[col].iloc[0], bool):
                 df[col] = df[col].astype(int)

    # Ensure target column is numeric
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in data.")
        return None
        
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        print(f"Error: Target column '{TARGET_COLUMN}' is not numeric. Check data.")
        return None

    print("Data preparation complete.")
    return df


def analyze_single_feature(df, feature_name):
    """
    Replicates the 'Quick & Dirty' Pivot Table analysis.
    Groups by a single feature and calculates the mortality rate and patient count.
    """
    if feature_name not in df.columns:
        print(f"Warning: Feature '{feature_name}' not found. Skipping.")
        return None
        
    print(f"\n--- Analysis for: {feature_name} ---")
    
    # Group by the feature, get the target, and aggregate
    analysis = df.groupby(feature_name)[TARGET_COLUMN].agg(
        Mortality_Rate='mean',
        Patient_Count='count'
    )
    
    # Format as percentage
    analysis['Mortality_Rate'] = (analysis['Mortality_Rate'] * 100).round(2)
    
    return analysis.sort_values(by='Mortality_Rate', ascending=False)


def analyze_age(df):
    """
    Replicates the age-based Pivot Table.
    Groups 'age_original' into 10-year bins.
    """
    if 'age_original' not in df.columns:
        print("Warning: 'age_original' column not found. Skipping age analysis.")
        return None
        
    print("\n--- Analysis for: Age (Binned) ---")
    
    # Create 10-year age bins
    # Bins from 0 to 110 (to include 100+)
    bins = list(range(0, 111, 10))
    labels = [f"{i}-{i+9}" for i in bins[:-1]]
    
    df_temp = df.copy()
    df_temp['age_bin'] = pd.cut(df_temp['age_original'], bins=bins, labels=labels, right=False)
    
    analysis = df_temp.groupby('age_bin', observed=True)[TARGET_COLUMN].agg(
        Mortality_Rate='mean',
        Patient_Count='count'
    )
    
    # Format as percentage
    analysis['Mortality_Rate'] = (analysis['Mortality_Rate'] * 100).round(2)
    
    return analysis


def find_top_predictors(df):
    """
    Replicates the 'Hackathon Pro-Tip' analysis (the best one).
    - Finds all diagnosis/admission columns ('dx_*', 'adm_*').
    - Unpivots them.
    - Filters for only positive cases (where the patient has the diagnosis).
    - Groups by the diagnosis name and calculates mortality rate.
    """
    print("\n--- Analysis: Top Predictors (from 'dx_' and 'adm_' columns) ---")
    
    # 1. Identify ID columns and Value columns (our predictors)
    id_vars = ['patient_id', 'visit_id', TARGET_COLUMN]
    
    # Dynamically find all 'dx_' and 'adm_' columns
    predictor_cols = [col for col in df.columns if col.startswith('dx_') or col.startswith('adm_')]
    
    if not predictor_cols:
        print("Warning: No 'dx_*' or 'adm_*' columns found. Skipping top predictor analysis.")
        return None

    print(f"Found {len(predictor_cols)} predictor columns to analyze.")

    # 2. Unpivot (Melt) the dataframe
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=predictor_cols,
        var_name='Diagnosis_Feature',
        value_name='Has_Diagnosis'
    )
    
    # 3. Filter for only positive cases (where Has_Diagnosis == 1)
    # This is *much* more efficient than grouping on 0s too.
    df_positive_cases = df_melted[df_melted['Has_Diagnosis'] == 1].copy()
    
    if df_positive_cases.empty:
        print("Warning: No positive cases found after unpivoting. Check your 'dx_ / adm_' columns.")
        return None

    # 4. Group by the feature name and aggregate
    analysis = df_positive_cases.groupby('Diagnosis_Feature')[TARGET_COLUMN].agg(
        Mortality_Rate='mean',
        Patient_Count='count'
    )
    
    # 5. Sort to find the most impactful features
    analysis_sorted = analysis.sort_values(by='Mortality_Rate', ascending=False)
    
    # Format as percentage
    analysis_sorted['Mortality_Rate'] = (analysis_sorted['Mortality_Rate'] * 100).round(2)
    
    return analysis_sorted


def statistical_analysis(df):
    """
    Provides comprehensive statistical analysis of the dataset.
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS REPORT")
    print("="*80)
    
    # --- 1. Dataset Overview ---
    print("\n--- 1. DATASET OVERVIEW ---")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # --- 2. Missing Data Analysis ---
    print("\n--- 2. MISSING DATA ANALYSIS ---")
    missing_stats = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values(
        by='Missing_Percentage', ascending=False
    )
    
    if len(missing_stats) > 0:
        print(f"\nColumns with Missing Data ({len(missing_stats)} total):")
        print(missing_stats.head(20))
    else:
        print("No missing data found!")
    
    # --- 3. Descriptive Statistics for Numerical Columns ---
    print("\n--- 3. DESCRIPTIVE STATISTICS (Numerical Columns) ---")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        print(f"\nFound {len(numerical_cols)} numerical columns:")
        
        # Basic stats
        stats = df[numerical_cols].describe().T
        stats['variance'] = df[numerical_cols].var()
        stats['skewness'] = df[numerical_cols].skew()
        stats['kurtosis'] = df[numerical_cols].kurtosis()
        
        # Reorder columns for better readability
        stats = stats[['count', 'mean', 'std', 'variance', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']]
        
        print("\n", stats.round(3))
    else:
        print("No numerical columns found.")
    
    # --- 4. Target Variable Analysis ---
    if TARGET_COLUMN in df.columns:
        print(f"\n--- 4. TARGET VARIABLE ANALYSIS ({TARGET_COLUMN}) ---")
        target_stats = df[TARGET_COLUMN].describe()
        print(f"\nTarget Variable Statistics:")
        print(target_stats)
        
        if df[TARGET_COLUMN].nunique() <= 10:  # Categorical or binary
            print(f"\nTarget Variable Distribution:")
            value_counts = df[TARGET_COLUMN].value_counts()
            value_percentages = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2)
            
            target_dist = pd.DataFrame({
                'Count': value_counts,
                'Percentage': value_percentages
            })
            print(target_dist)
    
    # --- 5. Correlation Analysis ---
    print("\n--- 5. CORRELATION ANALYSIS WITH TARGET ---")
    if TARGET_COLUMN in df.columns and len(numerical_cols) > 1:
        correlations = df[numerical_cols].corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(
            key=abs, ascending=False
        )
        
        print(f"\nTop 20 Features Most Correlated with {TARGET_COLUMN}:")
        corr_df = pd.DataFrame({
            'Feature': correlations.index[:20],
            'Correlation': correlations.values[:20].round(4)
        })
        corr_df.index = range(1, len(corr_df) + 1)
        print(corr_df)
        
        # Highly correlated features (positive)
        high_pos_corr = correlations[correlations > 0.3]
        if len(high_pos_corr) > 0:
            print(f"\nFeatures with Strong Positive Correlation (>0.3): {len(high_pos_corr)}")
            print(high_pos_corr)
        
        # Highly correlated features (negative)
        high_neg_corr = correlations[correlations < -0.3]
        if len(high_neg_corr) > 0:
            print(f"\nFeatures with Strong Negative Correlation (<-0.3): {len(high_neg_corr)}")
            print(high_neg_corr)
    
    # --- 6. Categorical Variables Analysis ---
    print("\n--- 6. CATEGORICAL VARIABLES ANALYSIS ---")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) > 0:
        print(f"\nFound {len(categorical_cols)} categorical columns:")
        
        cat_summary = pd.DataFrame({
            'Column': categorical_cols,
            'Unique_Values': [df[col].nunique() for col in categorical_cols],
            'Most_Common': [df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None for col in categorical_cols],
            'Most_Common_Frequency': [df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0 for col in categorical_cols],
            'Most_Common_Pct': [(df[col].value_counts().iloc[0] / len(df) * 100).round(2) if len(df) > 0 else 0 for col in categorical_cols]
        })
        
        print("\n", cat_summary.to_string(index=False))
    else:
        print("No categorical columns found (or all converted to numeric).")
    
    # --- 7. Data Type Summary ---
    print("\n--- 7. DATA TYPE SUMMARY ---")
    dtype_summary = df.dtypes.value_counts()
    print(dtype_summary)
    
    # --- 8. Outlier Detection (IQR Method) ---
    print("\n--- 8. OUTLIER DETECTION (IQR Method) ---")
    if len(numerical_cols) > 0:
        outlier_counts = {}
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
        
        outlier_df = pd.DataFrame({
            'Column': outlier_counts.keys(),
            'Outlier_Count': outlier_counts.values(),
            'Outlier_Percentage': [round(count / len(df) * 100, 2) for count in outlier_counts.values()]
        }).sort_values(by='Outlier_Count', ascending=False)
        
        outlier_df = outlier_df[outlier_df['Outlier_Count'] > 0]
        
        if len(outlier_df) > 0:
            print(f"\nColumns with Outliers (Top 20):")
            print(outlier_df.head(20).to_string(index=False))
        else:
            print("No significant outliers detected.")
    
    print("\n" + "="*80)
    print("END OF STATISTICAL ANALYSIS")
    print("="*80 + "\n")


def main():
    """
    Main function to run the complete analysis pipeline.
    """
    df = load_and_prep_data(FILEPATH)
    
    if df is None:
        print("\nScript aborted due to data loading errors.")
        return

    # --- Run Statistical Analysis ---
    statistical_analysis(df)

    # --- Run 'Quick & Dirty' Analysis (Suggestion 2) ---
    # You can swap 'dx_palliative' for any other single feature
    print(analyze_single_feature(df, 'dx_palliative'))
    
    # --- Run Categorical Analysis (Suggestion 3) ---
    print(analyze_single_feature(df, 'service_group'))
    print(analyze_single_feature(df, 'living_status'))
    
    # --- Run Age Analysis (Suggestion 3) ---
    print(analyze_age(df))
    
    # --- Run 'Hackathon Pro-Tip' Analysis (Suggestion 4) ---
    top_predictors = find_top_predictors(df)
    
    if top_predictors is not None:
        print("\n--- Top 20 Most Impactful Diagnoses (Highest Mortality Rate) ---")
        print(top_predictors.head(20))
        
        # This is a crucial filter: find high-impact features that are
        # also common enough to be reliable.
        print("\n--- Top 20 Most Impactful Diagnoses (with > 100 patients) ---")
        reliable_predictors = top_predictors[top_predictors['Patient_Count'] > 100]
        print(reliable_predictors.head(20))


if __name__ == "__main__":
    main()