# failprint

[![PyPI](https://img.shields.io/pypi/v/etsi-failprint.svg)](https://pypi.org/project/etsi-failprint/)


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

## Future Contribution Ideas

We welcome community contributions! Here are some ideas for future enhancements:

- **Integration with additional MLOps platforms**  
  Extend compatibility to more tools (e.g., Kubeflow, ZenML, Flyte) to streamline diagnostics in diverse ML pipelines[5][6].

- **Advanced visualization dashboards**  
  Add interactive dashboards (e.g., via Streamlit or Dash) for exploring failure patterns and root causes visually.

- **Explainability integrations**  
  Incorporate explainability libraries (e.g., SHAP, LIME) to provide feature attribution for failure segments[4].

- **Support for unstructured data**  
  Enable analysis of failures in NLP and CV models by integrating embedding-based clustering and drift detection[4].

- **Plugin system for custom analyses**  
  Allow users to add custom scripts or modules for domain-specific failure analysis.

- **Expanded documentation and tutorials**  
  Add more real-world examples, troubleshooting guides, and video walkthroughs to help new users get started quickly[10].

## 🤝 Contributing

Contributions are welcome. If you have an idea for optimisation of existing features or improvement, please open a PR on GitHub.
Please refer to [CONTRIBUTING.md](https://github.com/etsi-ai/etsi-failprint/blob/main/CONTRIBUTING.md) for contribution guidelines and ensure

## 📄 License

This project is licensed under the BSD 2-Clause License.

<details>
<summary>📜 Click here to view full license</summary>

<br>
