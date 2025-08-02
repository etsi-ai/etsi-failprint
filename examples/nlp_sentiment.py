import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

try:
    from etsi.failprint import analyze_nlp
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from etsi.failprint import analyze_nlp


def run_newsgroups_example():
    """
    Demonstrates how to use analyze_nlp with a real-world text dataset.
    """
    print("--- Running NLP 20 Newsgroups Failure Example ---")

 
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
   
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    texts_test = newsgroups_test.data
    y_test = newsgroups_test.target

   
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultinomialNB()),
    ])
    

    pipeline.fit(newsgroups_train.data, newsgroups_train.target)
   
    y_pred = pipeline.predict(texts_test)

    print("\nAnalyzing failures... (This may take a moment)")
    report = analyze_nlp(
        texts=texts_test,
        y_true=y_test,
        y_pred=y_pred
    )

 
    print("\n--- Generated Report ---")
    print(report)


if __name__ == "__main__":
    run_newsgroups_example()