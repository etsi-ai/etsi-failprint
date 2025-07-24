# Getting Started with etsi-failprint

Welcome to the beginner’s guide for using etsi-failprint — an MLOps-first diagnostic tool that helps you find out why your machine learning models are failing.

This guide is for absolute beginners who are new to open source, Python packaging, or ML diagnostics tools.

## 1. Project Overview

**What is etsi-failprint?**

etsi-failprint is a root cause analysis tool that helps you debug and understand the failure patterns in your machine learning models. It is designed with an MLOps-first mindset and integrates well with tools like MLflow, DVC, Airflow, etc.

**Key Features:**
- ✅ Segments failed predictions based on feature values
- ✅ Clusters similar failure cases
- ✅ Generates detailed markdown reports
- ✅ Identifies drift, bias, or imbalance in data
- ✅ Easy to integrate with ML pipelines

**Use Cases:**
- Model debugging and evaluation
- Auditing and reporting failures
- Improving data quality
- Continuous monitoring in MLOps workflows

## 2. Prerequisites

Before getting started, make sure you have the following tools and knowledge:

**Required:**
- Python 3.8 or higher
- pip (Python package installer)
- Git (version control system)

**Recommended:**
- VS Code or any other modern code editor
- Basic familiarity with Python, pandas, and ML concepts

**How to Install Python (if not already installed):**
- Download Python from [python.org](https://www.python.org/downloads/)
- Install it and make sure to check "Add Python to PATH"
- Verify using:
  ```sh
  python --version
  ```

## 3. 🚧 Setting Up Your Environment

### a. Fork and Clone the Repository
```sh
git clone https://github.com/YOUR_USERNAME/etsi-failprint.git
cd etsi-failprint
```

### b. Create and Activate a Virtual Environment

**For Windows:**
```sh
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```sh
python3 -m venv venv
source venv/bin/activate
```

### c. Install Dependencies
```sh
pip install -e .
```
If you face errors:
- Ensure you're in the root folder
- Try upgrading pip: `pip install --upgrade pip`
- Install missing packages manually (like pandas, sklearn)

## ## Example Scripts: Prerequisites

To run the example scripts in the `examples/` folder (such as `titanic_workflow.py` and `iris_pipeline.py`), you need to install the following Python packages:

```sh
pip install pandas scikit-learn seaborn
```

**What these packages are for:**
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning models and utilities
- `seaborn`: Loading sample datasets (e.g., Titanic)

**How to install all at once:**
Open your terminal, activate your virtual environment, and run:
```sh
pip install pandas scikit-learn seaborn
```

**Troubleshooting:**
- If you get `ModuleNotFoundError`, make sure your virtual environment is activated and run the install command again.
- If you use Jupyter Notebook, you may need to restart the kernel after installing new packages.

**Ready to run examples?**
Once installed, you can run any script in the `examples/` folder:
```sh
python examples/titanic_workflow.py
python examples/iris_pipeline.py
```

**Want to use your own dataset?**

Here’s a minimal working example:
```python
import pandas as pd
from etsi.failprint import analyze

X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [10, 20, 30]})
y_true = pd.Series([1, 0, 1])
y_pred = pd.Series([1, 1, 0])

report = analyze(X, y_true, y_pred, output="markdown", cluster=True)
print(report)
```

## 5. 📃 Viewing and Understanding Reports

**Output Files:**
- `reports/failprint_report.md` — summary report
- `failprint.log` — log file with debugging info

**Example Output Snippet:**
```
**feature1**:
- `2` → 50.0% in failures (∆ +21.4%)
```
This means:
- Out of all failures, 50% occurred when feature1=2
- The symbol ∆ shows deviation from the total distribution

You can view `.md` files using:
- VS Code Markdown preview
- Typora or any Markdown viewer

## 6. 🏛️ Real-World Example: Titanic Dataset

Step-by-Step:
- Load the Titanic dataset from seaborn or CSV
- Train a simple model (e.g., LogisticRegression)
- Use `analyze()` on your X, y_true, y_pred
- Print and inspect the Markdown report

```python
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = sns.load_dataset("titanic").dropna()
X = data[["age", "fare", "pclass"]]
y = data["survived"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Analyze
from etsi.failprint import analyze
report = analyze(X_test, y_test, y_pred, output="markdown", cluster=True)
print(report)
```

## 7. Common Errors & Troubleshooting

**venv activation error:**
- Fix (Windows PowerShell):
  ```sh
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

**ModuleNotFoundError:**
- Run:
  ```sh
  pip install pandas numpy scikit-learn
  ```

**Report not generating?**
- Ensure your working directory is correct
- Check that `reports/` exists or is being created

## 8. Contribution Guide

Want to contribute your own examples or improvements?
- Fork the repo
- Create a new branch: `git checkout -b docs-update`
- Add your dataset examples or update this guide
- Commit and push your changes
- Open a Pull Request with a short description

**Example contribution:**
- Add a Titanic or Iris workflow
- Write an alternate dataset analysis
- Improve documentation, README, or code comments

## 9.  Additional Resources

- Official Repo: etsi-failprint on GitHub
- Python Docs: https://docs.python.org
- Markdown Preview Guide: Markdown Cheat Sheet

Thank you for reading and happy contributing!

Made with ❤️ by the open-source community.
