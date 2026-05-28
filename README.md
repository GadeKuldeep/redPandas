# redpandas-ml

![PyPI - Version](https://img.shields.io/pypi/v/redpandas-ml)
![Python Version](https://img.shields.io/pypi/pyversions/redpandas-ml)

Enterprise-grade ML data-preparation pipeline with integrated visual health reporting.

## Installation

```bash
pip install redpandas-ml
```

### 💡 Global vs. Virtual Environments (Recommendation)

When installing `redpandas-ml`, it automatically installs heavy data science libraries like `pandas`, `numpy`, `matplotlib`, and `scikit-learn`. 

**Option A: Virtual Environment (Standard Practice)**
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install redpandas-ml
```
* **Pros:** Keeps your project isolated. The heavy libraries won't conflict with other projects on your computer.
* **Cons:** You have to activate the environment every time you want to run your code.

**Option B: Global Installation (Beginner Friendly)**
```bash
# Just run this in a normal terminal without activating any environment
pip install redpandas-ml
```
* **Pros:** You can open any random folder, write `import redpandas as rp`, and it works instantly. No setup required! 
* **Cons:** The dependencies are installed system-wide.

**Our Recommendation:** If you are just starting out or want to use `redpandas-ml` as a quick utility tool across many random CSV files on your computer, **install it globally**. If you are building a production application, use a **virtual environment**.

**Note:** The package name on PyPI is `redpandas-ml`, but you import it as `redpandas` using the alias `rp`:
```python
import redpandas as rp
```

## Quick Start (Chain Usage)

```python
import redpandas as rp

# Full automated pipeline
X, y = (
    rp.Prep("messy_data.csv")
      .clean()
      .transform()
      .plot_report(target_column="label", save_path="report.html")
      .split_target("label")
)
```

## Configuration (`PrepConfig`)

You can fully customize the pipeline behavior using the new `PrepConfig` class:

```python
from redpandas.config import PrepConfig

config = PrepConfig(
    missing_threshold=0.80,
    outlier_method="iqr",
    scaling_method="robust",
    categorical_imputation="mode",
    auto_encode=True
)

prep = rp.Prep("messy_data.csv", config=config)
```

## Diagnostics & Health Scoring

`redpandas` automatically diagnoses datasets upon initialization:
```python
print(prep.diagnostics["health_score"])
print(prep.diagnostics["issues"])
print(prep.diagnostics["recommendations"])
```

## Data Cleaning (`.clean()`)

| Step | Action | Config Reference |
|------|--------|------------------|
| 1 | Remove exact duplicate rows | `remove_duplicates` |
| 2 | Drop empty/sparse columns | `missing_threshold` |
| 3 | Drop zero-variance columns | `remove_constant_columns` |
| 4 | Remove statistical outliers | `outlier_method` |

## Data Transformation (`.transform()`)

| Step | Action | Config Reference |
|------|--------|------------------|
| 1 | Impute numeric columns | `numerical_imputation` |
| 2 | Impute categorical columns | `categorical_imputation` |
| 3 | Scale numeric columns | `scaling_method` |
| 4 | One-hot encode categories | `auto_encode` |

## Pipeline Persistence

Save and load the entire pipeline state:
```python
prep.save_pipeline("pipeline.pkl")
loaded_prep = rp.Prep.load_pipeline("pipeline.pkl")
```

## Scikit-Learn Integration

Export the exact configuration of `redpandas` to a scikit-learn `ColumnTransformer`:
```python
sklearn_pipeline = prep.to_sklearn_pipeline()
# Use directly in GridSearchCV or other scikit-learn tools
```

## Visualization

The `.plot_report()` method generates a single self-contained HTML report with zero external dependencies. The report contains:
- Dataset Overview & Data Health Score
- Memory Usage & Recommendations
- Missing Value Analysis
- Feature Distributions
- Outliers (Box-Whisker)
- Correlation Matrix
- Class Balance

## Publishing a New Version

To publish to PyPI using GitHub Actions:
1. Update `version` in `pyproject.toml`
2. Create and push a new tag:
   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```
3. GitHub Actions handles the build and trusted PyPI publishing automatically.