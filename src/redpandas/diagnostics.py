import pandas as pd
import numpy as np

def diagnose(df: pd.DataFrame) -> dict:
    score = 100
    issues = []
    recommendations = []
    
    # Missing values
    missing_count = df.isna().sum().sum()
    if missing_count > 0:
        score -= min(20, (missing_count / df.size) * 100)
        issues.append("Missing values detected")
        recommendations.append("Imputation required (median for numeric, mode for categorical)")

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        score -= min(10, (dup_count / len(df)) * 50)
        issues.append("Duplicate rows detected")
        recommendations.append("Deduplication recommended")
        
    # Constant columns
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if const_cols:
        score -= min(10, len(const_cols) * 5)
        issues.append("Constant or zero-variance columns detected")
        recommendations.append("Drop zero-variance columns")
        
    # Outliers (IQR approximation)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        outliers_detected = False
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).any():
                outliers_detected = True
                break
        if outliers_detected:
            score -= 10
            issues.append("Extreme outliers detected in numerical columns")
            recommendations.append("Apply outlier removal using IQR")
            
    # Correlations
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        corr_arr = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_arr, 0)
        if (corr_arr > 0.95).any():
            score -= 10
            issues.append("Highly correlated features detected")
            recommendations.append("Consider feature selection or PCA to reduce multicollinearity")
            
    # Target Class Balance would need target knowledge, omitted for general df pass
    
    return {
        "health_score": max(0, int(score)),
        "issues": issues,
        "recommendations": recommendations
    }
