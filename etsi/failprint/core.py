import pandas as pd
from datetime import datetime
from .segmenter import segment_failures
from .cluster import cluster_failures
from .correlate import compute_drift_correlation
from .report import ReportWriter
from .counterfactuals import suggest_counterfactual


def analyze(X: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series,
            threshold: float = 0.05,
            cluster: bool = True,
            drift_scores: dict = None,
            output: str = "markdown",
            log_path: str = "failprint.log"):

    assert len(X) == len(y_true) == len(y_pred), "Data length mismatch."
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    X = X.reset_index(drop=True)

    failed_idx = y_true != y_pred
    failed_X = X[failed_idx]

    if output == "counterfactuals":
        print("\nCounterfactual Suggestions:")
        for idx, row in failed_X.iterrows():
            suggestion = suggest_counterfactual(row, X, y_true.name or "target")
            if suggestion:
                original_input = row.to_dict()
                original_input.pop(y_true.name, None)
                print(f"\nOriginal Input ({idx}): {original_input}")
                print(f"Suggested Change: {', '.join([f'{k} to {v}' for k,v in suggestion.items()])}")
                print("Prediction: Success (counterfactual)")
        return

    segments = segment_failures(X, failed_X, threshold=threshold)
    clusters = cluster_failures(failed_X) if cluster else None
    drift_corr = compute_drift_correlation(X, y_true, drift_scores) if drift_scores else {}

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
