import warnings
import pandas as pd
import numpy as np

class Prep:
    def __init__(self, data):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("data must be a string path or a pandas DataFrame")
        
        self.initial_shape = self.df.shape
        self.logs = {}
        self._optimize_memory()

    def _optimize_memory(self):
        for col in self.df.select_dtypes(include=['int', 'int64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
        for col in self.df.select_dtypes(include=['float', 'float64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')

    def clean(self, remove_outliers=True, outlier_threshold=3.0, missing_col_threshold=0.70):
        # Step 1: Drop exact duplicate rows
        initial_len = len(self.df)
        self.df = self.df.drop_duplicates()
        self.logs["duplicates_removed"] = initial_len - len(self.df)

        # Step 2: Drop columns with > missing_col_threshold fraction of NaN values
        thresh = int(len(self.df) * (1 - missing_col_threshold))
        initial_cols = len(self.df.columns)
        self.df = self.df.dropna(axis=1, thresh=thresh)
        self.logs["empty_columns_dropped"] = initial_cols - len(self.df.columns)

        # Step 3: Drop columns where nunique() <= 1 (zero-variance / constant)
        initial_cols = len(self.df.columns)
        self.df = self.df.loc[:, self.df.nunique() > 1]
        self.logs["zero_variance_columns_dropped"] = initial_cols - len(self.df.columns)

        # Step 4: Remove outliers
        self.logs["outliers_removed"] = 0
        if remove_outliers:
            for col in self.df.select_dtypes(include=[np.number]).columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lo = Q1 - outlier_threshold * IQR
                hi = Q3 + outlier_threshold * IQR
                
                outliers = ((self.df[col] < lo) | (self.df[col] > hi))
                self.logs["outliers_removed"] += outliers.sum()
                
                self.df = self.df[~outliers]

        return self

    def transform(self, numerical_strategy="median", scaling="robust", encode_categoricals=True):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Use select_dtypes(include=["str", "category", "object"]) for Pandas 3 and 2 compatibility
        cat_cols = self.df.select_dtypes(include=["str", "category", "object"]).columns

        # Log missing value counts BEFORE imputation
        self.logs["numeric_imputations"] = int(self.df[numeric_cols].isna().sum().sum())
        self.logs["categorical_imputations"] = int(self.df[cat_cols].isna().sum().sum())

        # Impute numeric cols
        if numerical_strategy == "median":
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif numerical_strategy == "mean":
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        # Impute categorical cols
        for col in cat_cols:
            if self.df[col].isna().any():
                mode_vals = self.df[col].mode()
                if not mode_vals.empty:
                    self.df[col] = self.df[col].fillna(mode_vals.iloc[0])
                else:
                    self.df[col] = self.df[col].fillna("Missing")

        # Scale numeric cols
        if scaling == "robust":
            for col in numeric_cols:
                median = self.df[col].median()
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR != 0:
                    self.df[col] = (self.df[col] - median) / IQR
                else:
                    self.df[col] = self.df[col] - median
        elif scaling == "minmax":
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                else:
                    self.df[col] = self.df[col] - min_val
        elif scaling == "none":
            pass

        # Encode categoricals
        if encode_categoricals and len(cat_cols) > 0:
            self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True, dtype=int)

        return self

    def plot_report(self, target_column=None, save_path="redpandas_report.html", open_browser=True):
        try:
            from redpandas.visualizer import generate_report
        except ImportError as e:
            warnings.warn("Visualization dependencies are missing. Install with: pip install matplotlib seaborn")
            return self
        
        generate_report(self.df, self.logs, self.initial_shape, target_column=target_column, save_path=save_path, open_browser=open_browser)
        return self

    def plot_distributions(self):
        try:
            from redpandas.visualizer import plot_distributions
            import matplotlib.pyplot as plt
            plot_distributions(self.df)
            plt.show()
        except ImportError:
            warnings.warn("Visualization dependencies are missing.")
        return self

    def plot_correlation(self, method="pearson"):
        try:
            from redpandas.visualizer import plot_correlation
            import matplotlib.pyplot as plt
            plot_correlation(self.df, method=method)
            plt.show()
        except ImportError:
            warnings.warn("Visualization dependencies are missing.")
        return self

    def split_target(self, target_column: str):
        if target_column not in self.df.columns:
            raise KeyError(f"Target column '{target_column}' not found in DataFrame.")
        
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Coerce residual bool columns in X to int (sklearn compatibility)
        bool_cols = X.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            X[bool_cols] = X[bool_cols].astype(int)
        
        # Print a formatted pipeline report
        print("="*40)
        print("REDPANDAS PIPELINE REPORT")
        print("="*40)
        print(f"Initial Shape: {self.initial_shape}")
        print(f"Final Shape:   {self.df.shape}")
        print("\n--- Cleaning ---")
        print(f"Duplicates removed: {self.logs.get('duplicates_removed', 0)}")
        print(f"Empty columns dropped: {self.logs.get('empty_columns_dropped', 0)}")
        print(f"Zero-variance columns dropped: {self.logs.get('zero_variance_columns_dropped', 0)}")
        print(f"Outliers removed: {self.logs.get('outliers_removed', 0)}")
        print("\n--- Transformation ---")
        print(f"Numeric imputations: {self.logs.get('numeric_imputations', 0)}")
        print(f"Categorical imputations: {self.logs.get('categorical_imputations', 0)}")
        print("\n--- Output ---")
        print(f"Features (X) shape: {X.shape}")
        print(f"Target (y) shape: {y.shape}")
        print("="*40)
        
        return X, y
