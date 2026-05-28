import os
import base64
import warnings
import webbrowser
from io import BytesIO
from datetime import datetime
import numpy as np

_SNS = False
try:
    import seaborn as sns
    _SNS = True
except ImportError:
    pass

def _require_mpl():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError as e:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib") from e

def _fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def _safe_num_cols(df, max_cols=12):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > max_cols:
        warnings.warn(f"DataFrame has {len(num_cols)} numeric columns, limiting visualization to first {max_cols}.")
        return num_cols[:max_cols]
    return num_cols

def plot_distributions(df, cols=None):
    _require_mpl()
    import matplotlib.pyplot as plt
    if cols is None:
        cols = _safe_num_cols(df)
    
    n = len(cols)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric columns", ha="center")
        return fig

    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    fig.suptitle("Feature Distributions", fontsize=14)
    axes = np.atleast_1d(axes).flatten()
    
    cmap = plt.get_cmap("tab10")
    
    for i, col in enumerate(cols):
        ax = axes[i]
        valid_data = df[col].dropna()
        if len(valid_data) == 0:
            ax.set_title(col)
            continue
            
        color = cmap(i % 10)
        ax.hist(valid_data, bins=30, density=True, alpha=0.35, color=color)
        
        if _SNS:
            sns.kdeplot(valid_data, ax=ax, color=color, warn_singular=False)
        else:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(valid_data)
                x_vals = np.linspace(valid_data.min(), valid_data.max(), 100)
                ax.plot(x_vals, kde(x_vals), color=color)
            except (ImportError, ValueError, np.linalg.LinAlgError):
                pass
                
        ax.set_title(col)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    return fig

def plot_correlation(df, method="pearson"):
    _require_mpl()
    import matplotlib.pyplot as plt
    
    num_cols = _safe_num_cols(df, max_cols=20)
    if len(num_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation plot")
        
    corr = df[num_cols].corr(method=method)
    n = len(num_cols)
    
    fig, ax = plt.subplots(figsize=(max(6, n*0.6), max(5, n*0.6)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=7)
                
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return fig

def plot_missing(df):
    _require_mpl()
    import matplotlib.pyplot as plt
    
    missing_pct = (df.isna().sum() / len(df)) * 100
    missing_pct = missing_pct.sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(missing_pct) * 0.3)))
    
    if missing_pct.max() == 0:
        ax.text(0.5, 0.5, "✓ No missing values detected", ha="center", va="center", fontsize=12, color="green")
        ax.axis("off")
        return fig
        
    colors = ["#e63946" if val > 30 else "#457b9d" for val in missing_pct]
    bars = ax.barh(missing_pct.index, missing_pct, color=colors)
    
    ax.axvline(30, color="gray", linestyle="--", alpha=0.7)
    ax.text(31, len(missing_pct)-1, "30% threshold", color="gray", va="center")
    
    ax.axvline(70, color="gray", linestyle="--", alpha=0.7)
    ax.text(71, len(missing_pct)-1, "70% threshold", color="gray", va="center")
    
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va="center", fontsize=8)
            
    ax.set_xlim(0, 105)
    ax.set_title("Missing Values Percentage")
    plt.tight_layout()
    return fig

def plot_outliers(df, cols=None):
    _require_mpl()
    import matplotlib.pyplot as plt
    
    if cols is None:
        cols = _safe_num_cols(df)
        
    n = len(cols)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric columns", ha="center")
        return fig

    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    fig.suptitle("Outlier Detection (Box-Whisker)", fontsize=14)
    axes = np.atleast_1d(axes).flatten()
    
    cmap = plt.get_cmap("Set2")
    
    for i, col in enumerate(cols):
        ax = axes[i]
        valid_data = df[col].dropna()
        if len(valid_data) == 0:
            ax.set_title(col)
            continue
            
        color = cmap(i % 8)
        boxprops = dict(facecolor=color, color='black')
        flierprops = dict(marker='o', markerfacecolor='#e63946', markersize=3, alpha=0.5, linestyle='none')
        
        ax.boxplot(valid_data, vert=False, boxprops=boxprops, flierprops=flierprops, patch_artist=True)
        median_val = valid_data.median()
        ax.text(median_val, 1.1, f"{median_val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(col)
        ax.set_yticks([])
        
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    return fig

def plot_class_balance(df, target_column):
    _require_mpl()
    import matplotlib.pyplot as plt
    
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found.")
        
    counts = df[target_column].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Class Balance: {target_column}")
    
    cmap = plt.get_cmap("Pastel1")
    colors = [cmap(i % 9) for i in range(len(counts))]
    
    bars = ax1.bar(counts.index.astype(str), counts.values, color=colors)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height, f"{height}", ha="center", va="bottom")
    ax1.set_title("Counts")
    
    ax2.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", colors=colors)
    ax2.set_title("Percentages")
    
    plt.tight_layout()
    return fig

def generate_report(df, logs, initial_shape, diagnostics, *, target_column=None, save_path="redpandas_report.html", open_browser=True):
    _require_mpl()
    import matplotlib.pyplot as plt
    
    panels = []
    
    def _add_panel(title, func, *args, **kwargs):
        try:
            fig = func(*args, **kwargs)
            img_b64 = _fig_to_base64(fig)
            plt.close(fig)
            panels.append({"title": title, "img": img_b64})
        except Exception as e:
            warnings.warn(f"Failed to generate {title} plot: {e}")
            
    _add_panel("Missing Value Analysis", plot_missing, df)
    _add_panel("Feature Distribution Analysis", plot_distributions, df)
    _add_panel("Outlier Summary", plot_outliers, df)
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) >= 2:
        _add_panel("Correlation Analysis", plot_correlation, df)
        
    if target_column and target_column in df.columns:
        _add_panel("Class Balance", plot_class_balance, df, target_column)
        
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)

    health_score = diagnostics.get("health_score", 100)
    health_color = "#2ecc71" if health_score >= 80 else "#f1c40f" if health_score >= 60 else "#e74c3c"
    
    issues_html = "<ul>" + "".join(f"<li>{i}</li>" for i in diagnostics.get("issues", [])) + "</ul>" if diagnostics.get("issues") else "<p>No issues detected.</p>"
    recs_html = "<ul>" + "".join(f"<li>{r}</li>" for r in diagnostics.get("recommendations", [])) + "</ul>" if diagnostics.get("recommendations") else "<p>No recommendations.</p>"

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>redpandas Data Health Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f5f6fa; color: #2c3e50; }}
            .header {{ background-color: #1a1a2e; color: white; padding: 20px; display: flex; align-items: center; justify-content: space-between; }}
            .badge {{ background-color: #e63946; color: white; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }}
            .container {{ padding: 20px; max-width: 1400px; margin: 0 auto; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; border: 1px solid #e1e4e8; }}
            .stats-table {{ width: 100%; border-collapse: collapse; }}
            .stats-table th, .stats-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
            .stats-table th {{ color: #666; font-weight: 600; width: 40%; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 20px; }}
            .panel {{ background: white; border: 1px solid #e1e4e8; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            .panel-title {{ font-size: 1.1em; font-weight: 600; margin-top: 0; margin-bottom: 15px; color: #2c3e50; border-bottom: 2px solid #f0f2f5; padding-bottom: 10px; }}
            .panel img {{ width: 100%; height: auto; }}
            .health-score {{ font-size: 3em; font-weight: bold; color: {health_color}; margin: 10px 0; }}
            .footer {{ text-align: center; padding: 20px; color: #7f8c8d; font-size: 0.9em; margin-top: 40px; }}
            .flex-row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .flex-col {{ flex: 1; min-width: 300px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1 style="margin:0;">redpandas</h1>
            <div class="badge">DATA HEALTH REPORT</div>
        </div>
        <div class="container">
            
            <div class="flex-row">
                <div class="card flex-col">
                    <h2 style="margin-top:0;">Data Health Score</h2>
                    <div class="health-score">{health_score}/100</div>
                    
                    <h3>Issues Detected</h3>
                    {issues_html}
                    
                    <h3>Recommendations</h3>
                    {recs_html}
                </div>
                
                <div class="card flex-col">
                    <h2 style="margin-top:0;">Dataset Overview</h2>
                    <table class="stats-table">
                        <tr><th>Initial Shape</th><td>{initial_shape}</td></tr>
                        <tr><th>Final Shape</th><td>{df.shape}</td></tr>
                        <tr><th>Memory Usage</th><td>{memory_usage:.2f} MB</td></tr>
                        <tr><th>Duplicates Removed</th><td>{logs.get('duplicates_removed', 0)}</td></tr>
                        <tr><th>Empty Columns Dropped</th><td>{logs.get('empty_columns_dropped', 0)}</td></tr>
                        <tr><th>Zero Variance Columns Dropped</th><td>{logs.get('zero_variance_columns_dropped', 0)}</td></tr>
                        <tr><th>Outliers Removed</th><td>{logs.get('outliers_removed', 0)}</td></tr>
                        <tr><th>Numeric Imputations</th><td>{logs.get('numeric_imputations', 0)}</td></tr>
                        <tr><th>Categorical Imputations</th><td>{logs.get('categorical_imputations', 0)}</td></tr>
                    </table>
                </div>
            </div>
            
            <div class="grid">
                {''.join(f'<div class="panel"><h3 class="panel-title">{p["title"]}</h3><img src="data:image/png;base64,{p["img"]}" /></div>' for p in panels)}
            </div>
        </div>
        <div class="footer">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by redpandas-prep
        </div>
    </body>
    </html>
    """
    
    abs_path = os.path.abspath(save_path)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    print(f"Report saved to: {abs_path}")
    
    if open_browser:
        webbrowser.open("file://" + abs_path)
        
    return abs_path
