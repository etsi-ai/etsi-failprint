import pandas as pd
from etsi.failprint import analyze

X = pd.DataFrame({
    "feature1": [1, 2, 2, 3, 3, 3, 4],
    "feature2": [10, 15, 14, 13, 12, 13, 20],
    "category": ["A", "B", "B", "B", "C", "C", "A"]
})
y_true = pd.Series([1, 1, 1, 0, 0, 1, 0])
y_pred = pd.Series([1, 1, 0, 0, 0, 1, 1])

report = analyze(X, y_true, y_pred, output="markdown", cluster=True)
print(report)
