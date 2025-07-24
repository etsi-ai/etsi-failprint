from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from etsi.failprint import analyze
import pandas as pd

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Pipeline and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)  # Fix: Use X_test, not y_pred
y_pred = pd.Series(y_pred)         # Fix: Convert numpy array to pandas Series

# Analyze
report = analyze(X_test, y_test, y_pred, output="markdown", cluster=True)
print(report)

