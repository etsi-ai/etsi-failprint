import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from etsi.failprint import analyze

# Load dataset
data = sns.load_dataset("titanic").dropna()
X = data[["age", "fare", "pclass"]]
y = data["survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred)  # Fix: Convert numpy array to pandas Series

# Analyze
report = analyze(X_test, y_test, y_pred, output="markdown", cluster=True)
print(report)
