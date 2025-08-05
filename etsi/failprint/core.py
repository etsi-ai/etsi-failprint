import pandas as pd
from datetime import datetime
from failprint.nlp_features import build_nlp_feature_df
from .segmenter import segment_failures
from .cluster import cluster_failures
from .correlate import compute_drift_correlation
from .report import ReportWriter
from .nlp import cluster_failures_with_dbscan 
from .report import NlpReportWriter 

def analyze_nlp(texts: list, y_true: list, y_pred: list,
                model_name: str = "nlp_model",
                embedding_method: str = "sentence-transformers",
                cluster_failures: bool = True,
                output: str = "markdown",
                log_path: str = "failprint.log"):
    """
    Analyzes failures in NLP model predictions by segmenting on text 
    characteristics and clustering on semantic meaning.
    """
    assert len(texts) == len(y_true) == len(y_pred), "Data length mismatch."

    # Step 1: Create initial DataFrame and identify failures
    df = pd.DataFrame({
        'text': texts,
        'y_true': y_true,
        'y_pred': y_pred
    })
    failed_idx = df['y_true'] != df['y_pred']
    failures_df = df[failed_idx].copy()
    
    if failures_df.empty:
        print("[failprint] No failures detected. No report generated.")
        return "No failures detected."

    # --- Step 2: Extract NLP features for ALL texts ---
    # We use all texts to get a baseline distribution for comparison.
    # Pass df['text'] which is a pandas Series.
    nlp_features_df = build_nlp_feature_df(df['text'])
    
    # Isolate the features corresponding to the failed samples
    failed_nlp_features_df = nlp_features_df[failed_idx]

    # ---Step 3: Segment failures based on the extracted features ---
    # Reuse the same segmenter from the structured data analysis!
    nlp_segments = segment_failures(
        nlp_features_df, 
        failed_nlp_features_df, 
        threshold=0.05
    )

    # Step 4: Cluster failure cases by semantic meaning (if requested)
    clustered_failures_df = None
    if cluster_failures:
        # This function is from your nlp.py file
        clustered_failures_df = cluster_failures_with_dbscan(failures_df)

    # Step 5: Write the enhanced markdown report
    report_writer = NlpReportWriter(
        clustered_failures=clustered_failures_df,
        nlp_segments=nlp_segments, # --- NEW: Pass the segments to the writer
        output=output,
        log_path=log_path,
        failures=len(failures_df),
        total=len(df),
        timestamp=datetime.now().isoformat()
    )

    # Step 6: Generate and return the report
    return report_writer.write()

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
