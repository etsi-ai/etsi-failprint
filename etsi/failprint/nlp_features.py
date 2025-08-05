# failprint/nlp_features.py

import pandas as pd
import spacy
from textblob import TextBlob

# Load the spaCy model once to be efficient.
# It includes a user-friendly downloader if the model is not found.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm' (this may take a moment)...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("Model downloaded successfully.")

def extract_text_length(texts: pd.Series) -> pd.Series:
    """Calculates the character length of each text."""
    return texts.str.len()

def extract_word_count(texts: pd.Series) -> pd.Series:
    """Calculates the word count of each text."""
    return texts.str.split().str.len()

def extract_sentiment(texts: pd.Series) -> pd.DataFrame:
    """Calculates polarity and subjectivity for each text using TextBlob."""
    sentiments = texts.apply(lambda text: TextBlob(text).sentiment)
    # Ensure the output DataFrame has the same index as the input Series
    return pd.DataFrame(sentiments.tolist(), index=texts.index)

def extract_ner_counts(texts: pd.Series) -> pd.DataFrame:
    """Counts named entities (Person, Organization, Location) in each text."""
    ner_counts = []
    # Use nlp.pipe for efficiency on multiple texts
    for doc in nlp.pipe(texts):
        counts = {'PERSON_count': 0, 'ORG_count': 0, 'GPE_count': 0} # GPE: Geopolitical Entity
        for ent in doc.ents:
            label = f"{ent.label_}_count"
            if label in counts:
                counts[label] += 1
        ner_counts.append(counts)
    # Ensure the output DataFrame has the same index as the input Series
    return pd.DataFrame(ner_counts, index=texts.index)

def build_nlp_feature_df(texts: pd.Series) -> pd.DataFrame:
    """
    Creates a DataFrame of NLP features from a Series of texts.
    """
    # Basic features
    features_df = pd.DataFrame({
        'text_length': extract_text_length(texts),
        'word_count': extract_word_count(texts)
    }, index=texts.index)
    
    # Sentiment features
    sentiment_df = extract_sentiment(texts)
    
    # NER features
    ner_df = extract_ner_counts(texts)
    
    # Combine all features into a single DataFrame
    return pd.concat([features_df, sentiment_df, ner_df], axis=1)