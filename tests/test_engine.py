import pytest
import numpy as np
import pandas as pd
from redpandas.engine import Prep

@pytest.fixture
def messy_df():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "age": rng.integers(18, 80, size=n).astype(float),
        "income": rng.normal(50000, 15000, size=n),
        "score": rng.uniform(0, 100, size=n),
        "constant": np.ones(n),
        "category": rng.choice(["A", "B", "C"], size=n),
        "target": rng.choice([0, 1], size=n)
    })
    
    # inject duplicates
    df = pd.concat([df.iloc[:10], df]).reset_index(drop=True)
    
    # inject 20 NaNs into age
    nan_indices = rng.choice(len(df), size=20, replace=False)
    df.loc[nan_indices, "age"] = np.nan
    
    # inject one extreme outlier
    df.loc[0, "income"] = 999_999
    
    return df

def test_init_dataframe(messy_df):
    prep = Prep(messy_df)
    assert prep.initial_shape == messy_df.shape
    assert prep.df.shape == messy_df.shape

def test_clean_removes_duplicates(messy_df):
    prep = Prep(messy_df).clean(remove_outliers=False)
    assert prep.logs["duplicates_removed"] >= 9

def test_clean_drops_zero_variance(messy_df):
    prep = Prep(messy_df).clean(remove_outliers=False)
    assert "constant" not in prep.df.columns
    assert prep.logs["zero_variance_columns_dropped"] >= 1

def test_clean_removes_outliers(messy_df):
    prep = Prep(messy_df).clean(remove_outliers=True)
    assert prep.logs["outliers_removed"] >= 1

def test_transform_no_nulls(messy_df):
    prep = Prep(messy_df).clean().transform()
    assert prep.df.isna().sum().sum() == 0

def test_transform_encodes_categoricals(messy_df):
    prep = Prep(messy_df).clean().transform(encode_categoricals=True)
    assert "category" not in prep.df.columns
    # Check if encoded columns are present
    assert any(col.startswith("category_") for col in prep.df.columns)

def test_split_target_shapes(messy_df):
    prep = Prep(messy_df).clean().transform()
    X, y = prep.split_target("target")
    assert "target" not in X.columns
    assert len(X) == len(y)

def test_split_target_bad_column(messy_df):
    prep = Prep(messy_df).clean().transform()
    with pytest.raises(KeyError):
        prep.split_target("nonexistent_column")

def test_chain_returns_prep(messy_df):
    prep = Prep(messy_df)
    result = prep.clean().transform()
    assert isinstance(result, Prep)

def test_no_bool_columns_in_X(messy_df):
    prep = Prep(messy_df).clean().transform(encode_categoricals=True)
    X, y = prep.split_target("target")
    bool_cols = X.select_dtypes(include=['bool']).columns
    assert len(bool_cols) == 0
