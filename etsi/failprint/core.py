import pandas as pd
from datetime import datetime
from .segmenter import segment_failures
from .cluster import cluster_failures
from .correlate import compute_drift_correlation
from .report import ReportWriter


def analyze(X: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series,
            threshold: float = 0.05,
            cluster: bool = True,
            drift_scores: dict = None,
            output: str = "markdown",
            log_path: str = "failprint.log"):

    assert len(X) == len(y_true) == len(y_pred), "Data length mismatch."

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

    # Step 5: Write markdown report + logs
    report = ReportWriter(
        segments=segments,
        drift_map=drift_corr,
        clustered_segments=clusters,
        output=output,
        log_path=log_path,
        failures=len(failed_X),
        total=len(y_true),
        timestamp=datetime.now().isoformat()
    )

    return report.write()
