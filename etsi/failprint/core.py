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
            log_path: str = "failprint.log") -> str:
    
    assert len(X) == len(y_true) == len(y_pred), "Data length mismatch."

    failed_idx = y_true != y_pred
    failed_X = X[failed_idx]

    segments = segment_failures(X, failed_X, threshold=threshold)
    clustered = cluster_failures(failed_X) if cluster else None
    drift_map = compute_drift_correlation(X, y_true, drift_scores) if drift_scores else {}

    report = ReportWriter(
        segments=segments,
        drift_map=drift_map,
        clustered_segments=clustered,
        output=output,
        log_path=log_path,
        total=len(y_true),
        failures=failed_idx.sum(),
        timestamp=datetime.now().isoformat()
    )

    return report.write()
