import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

"""
- Feature Importance Bar Charts: Rank features by their contribution to failures
"""

def feature_importance_bar_plot(X, y_true, y_pred, top_n_features=10, save_path=None):
    # Identify incorrect predictions and calculate correlation score for each numeric feature.
    wrong_predictions = (y_true != y_pred).astype(int)
    feature_scores = {}
    for column in X.columns:
        if X[column].dtype in ["float64", "int64", "float32", "int32"]:
            try:
                corr, _ = stats.pointbiserialr(wrong_predictions, X[column])
                feature_scores[column] = abs(corr) if not np.isnan(corr) else 0
            except:
                feature_scores[column] = 0

    # Sort features by importance score and select the top N for visualization.
    sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    top_features = sorted_features[:top_n_features]
    feature_names = [name for name, score in top_features]
    scores = [score for name, score in top_features]
    
    feature_names.reverse()
    scores.reverse()

    # Create a horizontal bar plot to display the top features and their scores.
    plt.figure(figsize=(10, 8))
    bars = plt.barh(feature_names, scores, color="skyblue")
    plt.xlabel("Importance Score (Point Biserial Correlation)")
    plt.ylabel("Features")
    plt.title("Top Features Contributing to Model Failures")
    plt.grid(axis="x", alpha=0.3)
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
        )
    plt.tight_layout()

    # Save the plot to the specified path or display it interactively.
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Feature importance plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def analyze_failures(X, y_true, y_pred, y_proba=None, save_plots=False, output_directory="plots/"):
    print("--- Starting Failure Analysis ---")
    feature_imp_path = None
    if save_plots:
        feature_imp_path = os.path.join(output_directory, "feature_importance.png")
    feature_importance_bar_plot(X, y_true, y_pred, save_path=feature_imp_path)
    print("--- Failure Analysis Complete ---")

def analyze(X, y_true, y_pred, visualize=True, plot_config=None):
    # Wrapper for analysis, using a plot_config dict to manage visualizations.
    if visualize:
        if plot_config is None:
            plot_config = {
                "save_plots": True,
                "plot_format": "png",
                "output_dir": "reports/plots/",
            }
        
        save_path = None
        if plot_config.get("save_plots"):
            output_dir = plot_config.get("output_dir", "reports/plots/")
            plot_format = plot_config.get("plot_format", "png")
            save_path = os.path.join(output_dir, f"feature_importance.{plot_format}")
        feature_importance_bar_plot(X, y_true, y_pred, save_path=save_path)
        
# Create sample DataFrame
X_sample = pd.DataFrame({
    'feature_A': np.random.rand(200) * 10,
    'feature_B': np.random.rand(200) * 5,
    'feature_C': np.random.randn(200),
    'feature_D_noise': np.random.rand(200) * 100,
    'category': np.random.choice(['Alpha', 'Beta', 'Gamma'], 200)
})

# Create true labels based on a simple rule
y_true_sample = (X_sample['feature_A'] + X_sample['feature_B'] > 7.5).astype(int)

# Simulate predictions with 30 errors
y_pred_sample = y_true_sample.copy()
error_indices = np.random.choice(X_sample.index, size=30, replace=False)
y_pred_sample.loc[error_indices] = 1 - y_pred_sample.loc[error_indices]

# Run the analysis to display the plot directly
analyze_failures(X_sample, y_true_sample, y_pred_sample, save_plots=False)