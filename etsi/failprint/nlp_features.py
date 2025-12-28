# etsi/failprint/nlp_features.py

import pandas as pd
from textblob import TextBlob
import sys

_nlp_model = None

def get_spacy_model():
    """Singleton to load spacy model only when needed."""
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model
    
    try:
        import spacy
        try:
            # Try loading the small English model
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            # Download if missing
            print("[failprint] Downloading spaCy model 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp_model = spacy.load("en_core_web_sm")
        return _nlp_model
    except Exception as e:
        # Catch compatibility errors (like Pydantic/Python 3.14 issues)
        print(f"[failprint] Warning: Could not load spaCy ({e}). NER features will be disabled.")
        return None

def extract_text_length(texts: pd.Series) -> pd.Series:
    """Calculates the character length of each text."""
    return texts.str.len()

def extract_word_count(texts: pd.Series) -> pd.Series:
    """Calculates the word count of each text."""
    return texts.str.split().str.len()

def extract_sentiment(texts: pd.Series) -> pd.DataFrame:
    # Ensure text is string to avoid TextBlob errors on NaN
    sentiments = texts.apply(lambda text: TextBlob(str(text)).sentiment)
    return pd.DataFrame(sentiments.tolist(), index=texts.index)

def extract_ner_counts(texts: pd.Series) -> pd.DataFrame:
    nlp = get_spacy_model() 
    
    # Fallback if SpaCy failed to load (Graceful Degradation)
    if nlp is None:
        # Return empty counts so the analysis pipeline doesn't crash
        zeros = [0] * len(texts)
        return pd.DataFrame({
            'PERSON_count': zeros, 
            'ORG_count': zeros, 
            'GPE_count': zeros
        }, index=texts.index)

    ner_counts = []
    
    # Handle potential non-string inputs gracefully
    clean_texts = texts.fillna("").astype(str)
    
    # Use nlp.pipe for efficiency
    try:
        for doc in nlp.pipe(clean_texts):
            counts = {'PERSON_count': 0, 'ORG_count': 0, 'GPE_count': 0}
            for ent in doc.ents:
                label = f"{ent.label_}_count"
                if label in counts:
                    counts[label] += 1
            ner_counts.append(counts)
    except Exception as e:
        print(f"[failprint] Warning: Error during NER processing ({e}). returning empty NER features.")
        return pd.DataFrame([{'PERSON_count': 0, 'ORG_count': 0, 'GPE_count': 0}] * len(texts), index=texts.index)

    return pd.DataFrame(ner_counts, index=texts.index)

def build_nlp_feature_df(texts: pd.Series) -> pd.DataFrame:
    """
    Creates a DataFrame of NLP features from a Series of texts.
    """
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