<div align="center">

# etsi-failprint

### Automated Diagnostics. Root Cause Analysis. Actionable Insights.

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/PyPI-v0.2.0-blueviolet)](https://pypi.org/project/etsi-failprint/)
[![Status](https://img.shields.io/badge/Status-Alpha-green)]()

> **Don't just measure accuracy. Understand failure.**

`etsi-failprint` is a diagnostic tool designed to answer the question: *"Why is my model failing?"* It automatically isolates failure patterns across Tabular, NLP, and Computer Vision workflows, generating human-readable reports that pinpoint the root cause of errors.

[Features](#key-features) • [Installation](#installation) • [Quickstart](#quickstart) • [Multi-Modal Analysis](#multi-modal-analysis) • [Counterfactuals](#counterfactuals)

</div>

---

## Why Failprint?

Standard metrics (Accuracy, F1) tell you *how often* you fail. Failprint tells you *why*.

* **Multi-Modal Native**: Seamlessly analyze failures in structured DataFrames, raw Text, or Image datasets using a unified API.
* **Automated Segmentation**: Automatically discovers weak spots (e.g., "Model fails 80% of the time when `Income < 50k`").
* **Lazy & Lightweight**: Heavy dependencies (Torch, SpaCy, Transformers) are **lazy-loaded**. If you only analyze tabular data, you never pay the memory cost of deep learning libraries.
* **Robust**: Built with graceful degradation. If an optional dependency is missing or incompatible, Failprint adapts instead of crashing.

---

## Key Features

* **Smart Segmentation**: Identifies feature ranges or categories where error rates are statistically anomalous.
* **Semantic Clustering**:
    * *NLP*: Groups failed texts by semantic meaning using Sentence Transformers.
    * *CV*: Clusters failed images using ResNet embeddings to find visual patterns.
* **Meta-Feature Extraction**:
    * *Text*: Analyzes failures by length, sentiment, subjectivity, and NER entities.
    * *Vision*: Analyzes failures by brightness, contrast, aspect ratio, and dimensions.
* **Counterfactuals**: Suggests minimal changes to input data that would flip a failure to a success.
* **Actionable Reporting**: Outputs detailed Markdown reports with visual insights directly to your workspace.

---

## Installation

### Prerequisites
* Python (3.8 or later)

### From PyPI
```bash
pip install etsi-failprint
```

### From Source
```bash
git clone https://github.com/etsi-failprint/etsi-failprint.git
cd etsi-failprint
pip install -e .
```

---

## Quickstart

### 1. Tabular Data Analysis

Identify which features are driving your model's mistakes.

```python
import pandas as pd
from etsi.failprint import analyze

# Load your data
df = pd.read_csv("loan_predictions.csv")
X = df.drop("target", axis=1)
y_true = df["target"]
y_pred = pd.Series([0, 1, 0, ...]) # Your model's predictions

# Run analysis
report = analyze(
    X, y_true, y_pred,
    cluster=True,       # Cluster similar failures?
    output="markdown"   # Generate 'failprint_report.md'
)

print(report)
```

Output Insight: "Segment Age < 25 contributes to 40% of all failures."

## Multi-Modal Analysis

Failprint isn't just for spreadsheets. It understands unstructured data too.

### 2. NLP Analysis (Text)

Lazy-loads spacy and sentence-transformers to find semantic and structural failure patterns.

```python
from etsi.failprint import analyze_nlp

texts = [
    "I love this product!", 
    "Terrible service, very slow.", 
    "Product is okay but arrived late."
]
y_true = [1, 0, 0] # Sentiment labels
y_pred = [1, 1, 0] # Model predictions (Error on index 1)

report = analyze_nlp(texts, y_true, y_pred)
```

Output Insight: "Failures are highly correlated with Sentiment Polarity < -0.5 and Word Count < 5."

### 3. CV Analysis (Images)

Lazy-loads torch and torchvision to find visual failure clusters (e.g., "Dark images" or "Blurry dogs").

```python
from etsi.failprint import analyze_cv

images = ["img1.jpg", "img2.jpg", "img3.jpg"]
y_true = [0, 1, 0]
y_pred = [0, 0, 0]

analyze_cv(images, y_true, y_pred)
```

Output Insight: "Cluster 0 (Dark Images) accounts for 60% of false negatives."

---

## Counterfactuals

Go beyond diagnostics. Ask "What should have happened?" This mode suggests the minimal change required to fix a prediction.

```python
from etsi.failprint import analyze

# Run in counterfactual mode
analyze(
    X, y_true, y_pred,
    output="counterfactuals"
)
```

Example Output:

```plaintext
Original Input: {'Age': 22, 'Income': 35000, 'Education': 'High School'}
Suggested Change: Education to 'Bachelor's'
Prediction: Success (Counterfactual)
```

---

## Contributing

Pull requests are welcome!

Please refer to [CONTRIBUTING.md](https://github.com/etsi-ai/etsi-failprint/blob/main/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](https://github.com/etsi-ai/etsi-failprint/blob/main/CODE_OF_CONDUCT.md) before submitting a Pull Request.

---

## Join the Community

Connect with the **etsi.ai** team and other contributors on our Discord.

[![Discord](https://img.shields.io/badge/Discord-Join%20the%20Server-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/invite/VCeY6H72rq)

---

## License

This project is distributed under the **BSD-2-Clause License**. See the [LICENSE](https://github.com/etsi-ai/etsi-failprint/blob/main/LICENSE) for details.

---

> Built with ❤️ by etsi.ai
