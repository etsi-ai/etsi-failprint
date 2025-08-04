import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from datetime import datetime
from .segmenter import segment_failures
from .cluster import cluster_failures
from .correlate import compute_drift_correlation
from .report import ReportWriter
from .explain import explain_failures


def analyze(X: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series,
            threshold: float = 0.05,
            cluster: bool = True,
            explain: bool = False,
            model=None,
            X_train=None,
            drift_scores: dict = None,
            output: str = "markdown",
            log_path: str = "failprint.log"):

    assert len(X) == len(y_true) == len(y_pred), "Data length mismatch."


    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, name=y_true.name)

    # Align indices
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Step 1: Identify failures
    failed_idx = y_true != y_pred
    failed_X = X[failed_idx]

    # Step 2: Segment failure patterns
    segments = segment_failures(X, failed_X, threshold=threshold)

    # Step 3: Cluster failure cases
    clusters = cluster_failures(failed_X) if cluster else None

    # Step 4: Optional drift correlation
    drift_corr = compute_drift_correlation(X, y_true, drift_scores) if drift_scores else {}

    # Step 5: SHAP analysis
    shap_summary = None
    if explain:
        if model is not None and X_train is not None:
            shap_summary = explain_failures(model, X_train, failed_X)

    # Step 6: Write markdown report + logs
    report = ReportWriter(
        segments=segments,
        drift_map=drift_corr,
        clustered_segments=clusters,
        shap_summary=shap_summary,
        output=output,
        log_path=log_path,
        failures=len(failed_X),
        total=len(y_true),
        timestamp=datetime.now().isoformat()
    )

    return report.write()
