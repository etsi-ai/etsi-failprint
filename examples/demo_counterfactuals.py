import pandas as pd
from etsi.failprint.core import analyze

# Sample data
X = pd.DataFrame([
    {"age": 25, "income": 35000, "education": "high school"},
    {"age": 30, "income": 45000, "education": "bachelor's"},
    {"age": 25, "income": 35000, "education": "bachelor's"},  
    {"age": 40, "income": 30000, "education": "high school"},
])

# Ground truth and model predictions
y_true = pd.Series([1, 1, 1, 1], name="target")  
y_pred = pd.Series([0, 1, 1, 1])  

# Run with counterfactuals output
analyze(X, y_true, y_pred, output="counterfactuals")
