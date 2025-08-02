import pandas as pd
from datetime import datetime
from .segmenter import segment_failures
from .cluster import cluster_failures
from .correlate import compute_drift_correlation
from .report import ReportWriter
from .nlp import cluster_failures_with_dbscan 
from .report import NlpReportWriter 

def analyze_nlp(texts: list, y_true: list, y_pred: list,
                output: str = "markdown",
                log_path: str = "failprint.log"):
    """
    Analyzes failures in NLP model predictions.
    """
    assert len(texts) == len(y_true) == len(y_pred), "Data length mismatch."

    # Step 1: Identify failures
    # Create a DataFrame to easily manage the data
    df = pd.DataFrame({
        'text': texts,
        'y_true': y_true,
        'y_pred': y_pred
    })
    failures_df = df[df['y_true'] != df['y_pred']].copy()

    # Step 2: Cluster failure cases using your new NLP function
    # This calls the DBSCAN logic from your nlp.py file
    clustered_failures_df = cluster_failures_with_dbscan(failures_df)

    # Step 3: Write markdown report using your new NLP Report Writer
    # This passes the clustered DataFrame to the new report writer
    report = NlpReportWriter(
        clustered_failures=clustered_failures_df,
        output=output,
        log_path=log_path,
        failures=len(failures_df),
        total=len(df),
        timestamp=datetime.now().isoformat()
    )

    # Step 4: Generate the report and return it
    return report.write()

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
