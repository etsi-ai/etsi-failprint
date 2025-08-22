import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

QUICK_TEST = True

try:
    from etsi.failprint import analyze_nlp
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from etsi.failprint import analyze_nlp


def run_newsgroups_example():
    """
    Demonstrates how to use analyze_nlp with a real-world text dataset,
    showing model evaluation and failure analysis.
    """
    print("--- Running NLP 20 Newsgroups Failure Example ---")
    print("Loading 20 Newsgroups dataset for 4 categories...")
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                          shuffle=True, random_state=42)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                         shuffle=True, random_state=42)

    if QUICK_TEST:
        print("\n** Running in QUICK_TEST mode on 300 samples. **\n")
        texts_test = newsgroups_test.data[:300]
        y_test = newsgroups_test.target[:300]
    else:
        print("\n** Running on the full test set. This may take a few minutes. **\n")
        texts_test = newsgroups_test.data
        y_test = newsgroups_test.target

    print("Training a Naive Bayes classifier...")
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english')),
        ('clf', MultinomialNB(alpha=0.1)),  # alpha=0.1 is a common choice
    ])

    pipeline.fit(newsgroups_train.data, newsgroups_train.target)
    y_pred = pipeline.predict(texts_test)

    print("\n--- Model Performance Evaluation ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=newsgroups_test.target_names))

    print("\n--- Analyzing Failures with failprint ---")
    print("This may take a moment as NLP features are extracted...")

    report = analyze_nlp(
        texts=texts_test,
        y_true=y_test,
        y_pred=y_pred,
        model_name="20_Newsgroups_Classifier",
        cluster_failures=True,
        output="markdown"
    )

    print("\n--- failprint Report ---")
    print(report)

    # Save report to a Markdown file
    with open("failprint_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved failprint report to failprint_report.md")


if __name__ == "__main__":
    run_newsgroups_example()
