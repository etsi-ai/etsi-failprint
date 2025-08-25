from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import DBSCAN
import pandas as pd
from .cv_embedder import generate_image_embeddings


def cluster_failures(failed_X: pd.DataFrame):
    num = failed_X.select_dtypes(include='number')
    if num.shape[0] < 2:
        return None
    k = min(2, len(num))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(num)
    return labels


def cluster_cv_failures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clusters image-based failures using DBSCAN on image embeddings.
    'df' must contain an 'image_path' column.
    """
    if df.empty:
        return df
    embeddings = generate_image_embeddings(df['image_path'].tolist())
    if embeddings.numel() == 0: 
        df['cluster'] = -1 
        return df

    dbscan = DBSCAN(eps=1.5, min_samples=2, metric='euclidean')
    labels = dbscan.fit_predict(embeddings)
    df['cluster'] = labels
    
    return df