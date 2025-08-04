from scipy.stats import pearsonr
import numpy as np

def compute_drift_correlation(X, y_true, drift_scores: dict):
    if not drift_scores:
        return {}

    corr = {}
    for feat, drift_val in drift_scores.items():
        if feat not in X.columns:
            continue

        x_col = X[feat]
        try:
            if np.issubdtype(x_col.dtype, np.number):
                # Convert to NumPy arrays to avoid type errors
                corr_val, _ = pearsonr(x_col.to_numpy(), y_true.to_numpy())
                corr[feat] = corr_val
            else:
                corr[feat] = None
        except Exception:
            corr[feat] = None

    return corr
