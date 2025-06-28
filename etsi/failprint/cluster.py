from sklearn.cluster import KMeans
import pandas as pd


def cluster_failures(failed_X: pd.DataFrame):
    num = failed_X.select_dtypes(include='number')
    if num.shape[0] < 2:
        return None
    k = min(2, len(num))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(num)
    return labels
