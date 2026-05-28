from dataclasses import dataclass

@dataclass
class PrepConfig:
    missing_threshold: float = 0.70
    outlier_method: str = "iqr"
    outlier_iqr_multiplier: float = 3.0
    scaling_method: str = "robust"
    categorical_imputation: str = "mode"
    numerical_imputation: str = "median"
    auto_encode: bool = True
    remove_duplicates: bool = True
    remove_constant_columns: bool = True
