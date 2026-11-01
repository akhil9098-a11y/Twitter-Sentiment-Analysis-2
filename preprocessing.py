import re
# import pandas as pd


def clean_tweet(tweet):
    """
    Lightweight preprocessing for traditional ML models
    such as TF-IDF + Logistic Regression.

    Keeps sentiment-relevant information while removing
    unnecessary Twitter noise.
    """

    if not isinstance(tweet, str):
        return ""

    tweet = tweet.lower()

    # Replace URLs with a token instead of deleting them completely
    tweet = re.sub(r'https?://\S+|www\.\S+', ' URL ', tweet)

    # Replace @mentions with a token
    tweet = re.sub(r'@\w+', ' USER ', tweet)

    # Remove hashtag symbol but keep the actual word
    tweet = re.sub(r'#(\w+)', r'\1', tweet)

    # Convert common contractions
    tweet = tweet.replace("can't", "can not")
    tweet = tweet.replace("won't", "will not")
    tweet = tweet.replace("n't", " not")

    # Keep only letters/numbers and basic spaces
    tweet = re.sub(r'[^a-z0-9\s]', ' ', tweet)

    # Remove extra whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    return tweet


def preprocess_data(df, text_column='text'):
    """
    Apply preprocessing to an entire DataFrame.
    """

    df = df.copy()

    df['cleaned_text'] = df[text_column].apply(clean_tweet)

    # Remove empty tweets
    df = df[df['cleaned_text'].str.strip() != '']

    return df