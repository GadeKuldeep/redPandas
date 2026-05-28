
import pandas as pd
import numpy as np

from .config import PrepConfig
from .logger import get_logger
from .diagnostics import diagnose

class Prep:
    def __init__(self, data, config: PrepConfig = None):
        if config is None:
            config = PrepConfig()
        self.config = config
        self.logger = get_logger()
        
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("data must be a string path or a pandas DataFrame")
        
        self.initial_shape = self.df.shape
        self.logs = {}
        self._optimize_memory()
        
        # Diagnostic analysis
        self.diagnostics = diagnose(self.df)

    def _optimize_memory(self):
        for col in self.df.select_dtypes(include=['int', 'int64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
        for col in self.df.select_dtypes(include=['float', 'float64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')

    def clean(self, remove_outliers=None, outlier_threshold=None, missing_col_threshold=None):
        # Fallbacks to config for backward compatibility
        rem_outliers = remove_outliers if remove_outliers is not None else True
        out_thresh = outlier_threshold if outlier_threshold is not None else self.config.outlier_iqr_multiplier
        miss_thresh = missing_col_threshold if missing_col_threshold is not None else self.config.missing_threshold
        
        # Step 1: Drop exact duplicate rows
        initial_len = len(self.df)
        if self.config.remove_duplicates:
            self.df = self.df.drop_duplicates()
        dups = initial_len - len(self.df)
        self.logs["duplicates_removed"] = dups
        if dups > 0:
            self.logger.info(f"Removed {dups} duplicates")

        # Step 2: Drop columns with too many missing values
        thresh = int(len(self.df) * (1 - miss_thresh))
        initial_cols = len(self.df.columns)
        self.df = self.df.dropna(axis=1, thresh=thresh)
        dropped_cols = initial_cols - len(self.df.columns)
        self.logs["empty_columns_dropped"] = dropped_cols
        if dropped_cols > 0:
            self.logger.info(f"Dropped {dropped_cols} columns exceeding missing threshold")

        # Step 3: Drop zero-variance columns
        initial_cols = len(self.df.columns)
        if self.config.remove_constant_columns:
            self.df = self.df.loc[:, self.df.nunique() > 1]
        zero_var = initial_cols - len(self.df.columns)
        self.logs["zero_variance_columns_dropped"] = zero_var
        if zero_var > 0:
            self.logger.info(f"Dropped {zero_var} zero-variance columns")

        # Step 4: Remove outliers
        self.logs["outliers_removed"] = 0
        if rem_outliers and self.config.outlier_method == "iqr":
            for col in self.df.select_dtypes(include=[np.number]).columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lo = Q1 - out_thresh * IQR
                hi = Q3 + out_thresh * IQR
                
                outliers = ((self.df[col] < lo) | (self.df[col] > hi))
                count = outliers.sum()
                self.logs["outliers_removed"] += count
                
                self.df = self.df[~outliers]
            if self.logs["outliers_removed"] > 0:
                self.logger.info(f"Removed {self.logs['outliers_removed']} outliers")

        return self

    def transform(self, numerical_strategy=None, scaling=None, encode_categoricals=None):
        num_strat = numerical_strategy if numerical_strategy is not None else self.config.numerical_imputation
        scl = scaling if scaling is not None else self.config.scaling_method
        enc = encode_categoricals if encode_categoricals is not None else self.config.auto_encode
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(include=["str", "category", "object"]).columns

        # Log missing value counts BEFORE imputation
        num_nan = int(self.df[numeric_cols].isna().sum().sum())
        cat_nan = int(self.df[cat_cols].isna().sum().sum())
        self.logs["numeric_imputations"] = num_nan
        self.logs["categorical_imputations"] = cat_nan
        
        if num_nan > 0:
            self.logger.info(f"Imputed {num_nan} missing values in numeric columns")
        if cat_nan > 0:
            self.logger.info(f"Imputed {cat_nan} missing values in categorical columns")

        # Impute numeric cols
        if num_strat == "median":
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif num_strat == "mean":
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        # Impute categorical cols
        cat_strat = self.config.categorical_imputation
        for col in cat_cols:
            if self.df[col].isna().any():
                if cat_strat == "mode":
                    mode_vals = self.df[col].mode()
                    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Missing"
                else:
                    fill_val = "Missing"
                self.df[col] = self.df[col].fillna(fill_val)

        # Scale numeric cols
        if scl == "robust":
            for col in numeric_cols:
                median = self.df[col].median()
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR != 0:
                    self.df[col] = (self.df[col] - median) / IQR
                else:
                    self.df[col] = self.df[col] - median
        elif scl == "minmax":
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                else:
                    self.df[col] = self.df[col] - min_val
        elif scl == "none":
            pass

        # Encode categoricals
        if enc and len(cat_cols) > 0:
            self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True, dtype=int)
            self.logger.info(f"One-hot encoded {len(cat_cols)} categorical columns")

        return self

    def plot_report(self, target_column=None, save_path="redpandas_report.html", open_browser=True):
        try:
            from .visualizer import generate_report
        except ImportError as e:
            self.logger.warning("Visualization dependencies are missing. Install with: pip install matplotlib seaborn")
            return self
        
        generate_report(self.df, self.logs, self.initial_shape, self.diagnostics, 
                        target_column=target_column, save_path=save_path, open_browser=open_browser)
        return self

    def plot_distributions(self):
        try:
            from .visualizer import plot_distributions
            import matplotlib.pyplot as plt
            plot_distributions(self.df)
            plt.show()
        except ImportError:
            self.logger.warning("Visualization dependencies are missing.")
        return self

    def plot_correlation(self, method="pearson"):
        try:
            from .visualizer import plot_correlation
            import matplotlib.pyplot as plt
            plot_correlation(self.df, method=method)
            plt.show()
        except ImportError:
            self.logger.warning("Visualization dependencies are missing.")
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
        
        self.logger.info("Pipeline complete. Splitting features and target.")
        
        # Print a formatted pipeline report
        print("="*40)
        print("REDPANDAS PIPELINE REPORT")
        print("="*40)
        print(f"Health Score:  {self.diagnostics['health_score']}/100")
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

    def save_pipeline(self, filepath: str):
        """Save the Prep instance using pickle."""
        from .persistence import save_pipeline
        save_pipeline(self, filepath)
        self.logger.info(f"Pipeline saved to {filepath}")

    @classmethod
    def load_pipeline(cls, filepath: str):
        """Load a Prep instance from a pickle file."""
        from .persistence import load_pipeline
        return load_pipeline(filepath)

    def to_sklearn_pipeline(self):
        """Export the preprocessing configuration as a scikit-learn pipeline."""
        from .sklearn_export import to_sklearn_pipeline
        return to_sklearn_pipeline(self)
