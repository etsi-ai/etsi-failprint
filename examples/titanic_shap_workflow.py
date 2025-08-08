import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from etsi.failprint import analyze

# 1. Load and preprocess data
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df.drop('Survived', axis=1)
y = df['Survived']

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Analyze with SHAP enabled
print("Running failprint analysis with SHAP...")
analyze(
    X=X_test, 
    y_true=y_test, 
    y_pred=y_pred, 
    model=model, 
    X_train=X_train, 
    explain=True
)
print("Analysis complete. Check the 'reports/failprint_report.md' file.")