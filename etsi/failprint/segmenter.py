import pandas as pd
from collections import defaultdict
from .utils import safe_div


def segment_failures(X: pd.DataFrame, failed_X: pd.DataFrame, threshold: float = 0.05):
    segments = defaultdict(list)

    for col in X.columns:
        for val in X[col].unique():
            full_pct = safe_div((X[col] == val).sum(), len(X))
            fail_pct = safe_div((failed_X[col] == val).sum(), len(failed_X))
            delta = fail_pct - full_pct

            if delta >= threshold:
                segments[col].append((val, fail_pct, delta))

    return segments
