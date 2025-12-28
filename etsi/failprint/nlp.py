# etsi/failprint/nlp.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def convert_to_embeddings(texts: list, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Converts a list of texts to embeddings."""
    # Lazy import
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

def cluster_failures_with_dbscan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clusters failures using DBSCAN and returns the DataFrame with a 'cluster' column.
    """
    if df.empty:
        return df
  
    embeddings = convert_to_embeddings(df['text'].tolist())
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    dbscan.fit(embeddings)
    df['cluster'] = dbscan.labels_
    
    return df