# failprint_demo.py

"""
This script demonstrates how to use `failprint` in a realistic ML workflow.
It simulates a binary classification problem, trains a model, and analyzes failures.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from etsi.failprint import analyze

# Simulated dataset
from sklearn.datasets import make_classification

# Step 1: Generate data
X_raw, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    weights=[0.7, 0.3],
    random_state=42
)

# Step 2: Wrap in DataFrame with semantic names
X = pd.DataFrame(X_raw, columns=[f"feature{i+1}" for i in range(X_raw.shape[1])])
X['user_segment'] = pd.qcut(X['feature1'], q=4, labels=['low', 'mid-low', 'mid-high', 'high'])  # add a categorical feature

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Step 4: Train a model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train.drop(columns=['user_segment']), y_train)

# Step 5: Make predictions
y_pred = clf.predict(X_test.drop(columns=['user_segment']))

# Step 6: Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"[model] Test Accuracy: {acc:.4f}")

# Step 7: Use failprint to analyze failures
print("\n[failprint] Analyzing model failures...\n")
report = analyze(
    X=X_test[['feature1', 'feature2', 'user_segment']],  # only select useful or explainable features
    y_true=pd.Series(y_test),
    y_pred=pd.Series(y_pred),
    output="markdown",
    cluster=True
)

# Display full markdown report
print(report)

# Optional: manually check logs/report
with open("failprint.log", "r", encoding="utf-8") as f:
    print("\n[failprint.log]")
    print(f.read())

with open("reports/failprint_report.md", "r", encoding="utf-8") as f:
    print("\n[failprint_report.md]")
    print(f.read())
