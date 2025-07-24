"""
Wine Quality Dataset + RandomForest + failprint Analysis
"""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from etsi.failprint import analyze

# Step 1: Load the Wine dataset
data = load_wine(as_frame=True)
X = data.data
y = data.target

# For simplicity, convert to binary classification (class 0 vs others)
y_binary = (y == 0).astype(int)

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Step 3: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred)  # Fix: Convert numpy array to pandas Series

# Step 5: Analyze with failprint
report = analyze(X_test, y_test, y_pred, output="markdown", cluster=True)

# Step 6: Print the report or save to file
print(report)

with open("reports/wine_failprint_report.md", "w") as f:
    f.write(report)
