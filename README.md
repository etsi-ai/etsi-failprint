# failprint

[![PyPI](https://img.shields.io/pypi/v/etsi-failprint.svg)](https://pypi.org/project/etsi-failprint/)
[![PyPI Downloads](https://static.pepy.tech/badge/etsi-watchdog)](https://pepy.tech/projects/etsi-watchdog)


**failprint** is an MLOps-first diagnostic tool that performs automatic root cause analysis on your ML model's failure patterns.

It segments, clusters, and correlates failed predictions with input data features — surfacing **which features are contributing to failure**, **which data segments fail the most**, and **how drift or imbalance may be related to model degradation**.

##  Installation

```bash
pip install etsi-failprint
```

##  Quick Start

``` bash 

import pandas as pd
from etsi.failprint import analyze

# Sample inputs
X = pd.DataFrame({
    "feature1": [1, 2, 2, 3, 3, 3, 4],
    "feature2": [10, 15, 14, 13, 12, 13, 20],
    "category": ["A", "B", "B", "B", "C", "C", "A"]
})
y_true = pd.Series([1, 1, 1, 0, 0, 1, 0])
y_pred = pd.Series([1, 1, 0, 0, 0, 1, 1])

# Analyze misclassifications
report = analyze(X, y_true, y_pred, output="markdown", cluster=True)
print(report)

```

##  What It Does
- Segments failures by input feature values (numerical/categorical)
- Highlights overrepresented values in failure cases
- Clusters similar failure samples for pattern recognition
- Writes log files and markdown reports for audit or CI/CD
- Compatible with MLOps tools (like MLflow, DVC, Airflow, Watchdog)
