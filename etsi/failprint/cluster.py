from sklearn.cluster import KMeans
import pandas as pd


def cluster_failures(failed_X: pd.DataFrame):
    num = failed_X.select_dtypes(include='number')
    if num.shape[0] < 2:
        return []
    k = min(2, len(num))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(num)
    # Group rows by cluster label and return as list of dicts for each cluster
    clusters = []
    for cluster_id in range(k):
        cluster_rows = failed_X[labels == cluster_id]
        clusters.append(cluster_rows.to_dict(orient='records'))
    return clusters
