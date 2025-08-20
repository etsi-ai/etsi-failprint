# failprint

[![PyPI](https://img.shields.io/pypi/v/etsi-failprint.svg)](https://pypi.org/project/etsi-failprint/)

**failprint** is an MLOps-first diagnostic tool that performs automatic root cause analysis on your ML model's failure patterns.

It segments, clusters, and correlates failed predictions with input data features — surfacing **which features are contributing to failure**, **which data segments fail the most**, and **how drift or imbalance may be related to model degradation**.

---

## 📑 Table of Contents

- [🚀 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [📊 What It Does](#-what-it-does)
- [📚 Documentation & Examples](#-documentation--examples)
- [🛠️ Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [💡 Future Contribution Ideas](#-future-contribution-ideas)
- [🗂️ License](#-license)

## 🛠️ Requirements

Before installing **failprint**, make sure your environment meets the following requirements:

- **Python 3.7 or above**
- **pip** (Python package installer)
- Internet connection to download dependencies

Optional (for examples and extended functionality):
- `pandas`
- `scikit-learn`
- `seaborn`

---
## 🚀 Installation

```sh
pip install etsi-failprint
```

---

## ⚡ Example Prerequisites

To run the example scripts, install these packages:

```sh
pip install pandas scikit-learn seaborn
```

---

## 🏁 Quick Start

```python
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

---

## 📊 What It Does

- Segments failures by input feature values (numerical/categorical)
- Highlights overrepresented values in failure cases
- Clusters similar failure samples for pattern recognition
- Writes log files and markdown reports for audit or CI/CD
- Compatible with MLOps tools (like MLflow, DVC, Airflow, Watchdog)

---

## 📚 Documentation

- [Getting Started Guide](docs/getting_started.md)
- [Example: Titanic Dataset](examples/titanic_workflow.py)
- [Example: Iris Dataset with Pipeline](examples/iris_pipeline.py)
- [Example: Wine Dataset with Random Forest](examples/wine_random_forest.py)

The example scripts will print a markdown report to your terminal and may also generate a file in the `reports/` folder.

---

## 🗂️ Project Structure

- The repository is organized as follows:

```text

failprint/
├── docs/              # Documentation and contributor guides
├── etsi/failprint/    # Core source code of the failprint package
├── examples/          # Example workflows and usage scripts
├── reports/           # Generated reports and analysis outputs
├── test/              # Unit tests and validation scripts
├── .gitignore         # Files and directories ignored by Git
├── CODE_OF_CONDUCT.md # Contributor code of conduct
├── CONTRIBUTING.md    # Guidelines for contributing
├── LICENSE            # License information
├── README.md          # Project overview and usage guide
├── failprint.log      # Log file for debugging and analysis
├── pyproject.toml     # Project dependencies and build configuration
└── setup.cfg          # Packaging and setup configuration

```
---

## 🛠️ Troubleshooting

- **Can’t activate venv?**  
  Try:  
  ```sh
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **ModuleNotFoundError?**  
  Make sure your virtual environment is activated and run:  
  ```sh
  pip install pandas scikit-learn seaborn
  ```
- **Report not generating?**  
  Ensure your working directory is correct and that the `reports/` folder exists.

---

## 🤝 Contributing

Contributions are welcome! If you have an idea for optimization, new features, or documentation improvements, please open a PR on GitHub.

- Refer to [CONTRIBUTING.md](https://github.com/etsi-ai/etsi-failprint/blob/main/CONTRIBUTING.md) for contribution guidelines.
- If you have a workflow, dataset, or troubleshooting tip to share, please contribute to our [Getting Started Guide](docs/getting_started.md)!

---

## 💡 Future Contribution Ideas

We welcome community contributions! Here are some ideas for future enhancements:

- **Integration with additional MLOps platforms**  
  Extend compatibility to more tools (e.g., Kubeflow, ZenML, Flyte) to streamline diagnostics in diverse ML pipelines.

- **Advanced visualization dashboards**  
  Add interactive dashboards (e.g., via Streamlit or Dash) for exploring failure patterns and root causes visually.

- **Explainability integrations**  
  Incorporate explainability libraries (e.g., SHAP, LIME) to provide feature attribution for failure segments.

- **Support for unstructured data**  
  Enable analysis of failures in NLP and CV models by integrating embedding-based clustering and drift detection.

- **Plugin system for custom analyses**  
  Allow users to add custom scripts or modules for domain-specific failure analysis.

- **Expanded documentation and tutorials**  
  Add more real-world examples, troubleshooting guides, and video walkthroughs to help new users get started quickly.

---

## 📬 Contact

If you have questions, feedback, or ideas, feel free to reach out to the maintainers:

- **Priyansh Srivastava** – [GitHub](https://github.com/PriyanshSrivastava0305) | [Email](mailto:priyansh0305@gmail.com)  
- **Romit Chatterjee** – [GitHub](https://github.com/Romit23) | [Email](mailto:chatterjeeromit86@gmail.com)  

We’d love to hear from you and make **failprint** better together!

---

## 📄 License

This project is licensed under the BSD 2-Clause License.

<details>
<summary>📜 Click here to view full license</summary>

<br>

    BSD 2-Clause License

    Copyright (c) 2025, et-si.ai

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
    AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

</details>

---

> Made with ❤️ by the etsi-ai team


