import numpy as np

def to_sklearn_pipeline(prep_instance):
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
    except ImportError:
        raise ImportError("scikit-learn is required for this feature. Install it with: pip install scikit-learn")

    config = prep_instance.config
    
    num_cols = prep_instance.df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = prep_instance.df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Numeric preprocessing
    num_strategy = config.numerical_imputation
    num_imputer = SimpleImputer(strategy=num_strategy)
    
    if config.scaling_method == "robust":
        scaler = RobustScaler()
    elif config.scaling_method == "minmax":
        scaler = MinMaxScaler()
    elif config.scaling_method == "standard":
        scaler = StandardScaler()
    else:
        scaler = "passthrough"
        
    if scaler == "passthrough":
        num_pipeline = Pipeline(steps=[('imputer', num_imputer)])
    else:
        num_pipeline = Pipeline(steps=[('imputer', num_imputer), ('scaler', scaler)])

    # Categorical preprocessing
    cat_strategy = 'most_frequent' if config.categorical_imputation == "mode" else "constant"
    fill_val = None if cat_strategy == 'most_frequent' else "Missing"
    cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_val)
    
    if config.auto_encode:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        cat_pipeline = Pipeline(steps=[('imputer', cat_imputer), ('encoder', encoder)])
    else:
        cat_pipeline = Pipeline(steps=[('imputer', cat_imputer)])
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor
