import pandas as pd
import numpy as np

def validate_data(X: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series):

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    if not isinstance(y_true, pd.Series):
        raise ValueError("y_true must be a pandas Series.")
    if not isinstance(y_pred, pd.Series):
        raise ValueError("y_pred must be a pandas Series.")

    if len(X) != len(y_true) or len(X) != len(y_pred):
        raise ValueError("length of X, y_true, y_pred must be the same.")

    if X.isnull().values.any():
        raise ValueError("Input features contains missing values.")
    if y_true.isnull().any() or y_pred.isnull().any():
        raise ValueError("True or predicted labels contain missing values.")
    
    return True