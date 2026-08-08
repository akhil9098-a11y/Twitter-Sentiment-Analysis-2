import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import preprocess_data
from twitter_client import TwitterClient

MODEL_PATH = "model.joblib"
API_DATA_PATH = "data/api_tweets.csv"

def train_and_save_model(data_path=None):
    """
    Trains a model on the provided dataset and saves it to disk.
    If data_path is None, it tries to load from data/tweets.csv and data/api_tweets.csv.
    """
    dfs = []
    
    # 1. Load Base Dataset
    base_data = "data/tweets.csv"
    if os.path.exists(base_data):
        print(f"Loading base data from {base_data}...")
        # Sentiment140 format: target, id, date, flag, user, text
        df_base = pd.read_csv(base_data, encoding='latin-1', header=None, names=['sentiment', 'id', 'date', 'flag', 'user', 'text'])
        dfs.append(df_base[['text', 'sentiment']])
    
    # 2. Load API Data
    if os.path.exists(API_DATA_PATH):
        print(f"Loading API data from {API_DATA_PATH}...")
        df_api = pd.read_csv(API_DATA_PATH)
        dfs.append(df_api[['text', 'sentiment']])
        
    if data_path and data_path not in [base_data, API_DATA_PATH]:
        print(f"Loading additional data from {data_path}...")
        # Try to guess format
        temp_df = pd.read_csv(data_path, encoding='latin-1')
        if 'text' in temp_df.columns and 'sentiment' in temp_df.columns:
            dfs.append(temp_df[['text', 'sentiment']])
        else:
            # Assume Sentiment140 format
            temp_df = pd.read_csv(data_path, encoding='latin-1', header=None, names=['sentiment', 'id', 'date', 'flag', 'user', 'text'])
            dfs.append(temp_df[['text', 'sentiment']])

    if not dfs:
        raise ValueError("No data found for training. Please provide a valid data_path or ensure data/tweets.csv exists.")

    df = pd.concat(dfs).drop_duplicates(subset=['text'])
    
    print("Preprocessing data...")
    df = preprocess_data(df, 'text')
    
    # Drop rows where text became empty after cleaning
    df = df[df['cleaned_text'].str.strip() != '']

    X = df['cleaned_text']
    y = df['sentiment']
    
    print(f"Training on {len(df)} samples...")
    # The pipeline vectorizes the text using TF-IDF and applies Logistic Regression.
    # Improved TF-IDF with sublinear scaling and frequency filtering
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1,2),
            sublinear_tf=True,
            min_df=5,
            max_df=0.9
        )),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])
    
    # If we have enough data, do train/test split.
    if len(df) > 50:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        pipeline.fit(X_train, y_train)
        
        print("Evaluating model...")
        predictions = pipeline.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, predictions)}")
        print(classification_report(y_test, predictions, zero_division=0))
    else:
        pipeline.fit(X, y)
        print(f"Dataset too small ({len(df)} samples) for train/test split. Trained on all data.")
        
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    print("Model saved successfully.")

def collect_api_data_and_train(limit=100):
    """
    Fetches data from Twitter API, saves it, and retrains the model.
    """
    client = TwitterClient()
    new_data = client.get_labeled_data(limit_per_sentiment=limit)
    
    if not new_data:
        print("No data fetched from Twitter. Check API credentials or limits.")
        return False

    new_df = pd.DataFrame(new_data)
    
    # Prepend dummy columns to match Sentiment140 format if needed, 
    # but our train function is flexible enough if we adjust it.
    # Let's save it to a separate file and then combine.
    
    if os.path.exists(API_DATA_PATH):
        existing_api_df = pd.read_csv(API_DATA_PATH)
        combined_df = pd.concat([existing_api_df, new_df]).drop_duplicates(subset=['text'])
        combined_df.to_csv(API_DATA_PATH, index=False)
    else:
        new_df.to_csv(API_DATA_PATH, index=False)
    
    print(f"New data saved to {API_DATA_PATH}. Total API samples: {len(pd.read_csv(API_DATA_PATH))}")
    
    # Combine with original training data if available
    base_data = "data/tweets.csv"
    if os.path.exists(base_data):
        print("Combining with base dataset...")
        # Since the base dataset is large, we might want to just train on a sample + new API data
        # or just train on API data for fine-tuning (though LR doesn't fine-tune easily without full retrain).
        # For simplicity, we'll retrain on a combination.
        train_and_save_model(base_data) # This currently ignores API_DATA_PATH, let's fix that.
    else:
        # Train only on API data
        train_and_save_model(API_DATA_PATH)
    
    return True

def load_model():
    """
    Loads the saved model from disk.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please train the model first.")
    return joblib.load(MODEL_PATH)

def predict_sentiment(text, model):
    """
    Predicts sentiment for a single text input.
    """
    from preprocessing import clean_tweet
    cleaned_text = clean_tweet(text)
    if not cleaned_text:
        return "Unknown (Text too short after cleaning)"
    
    prediction = model.predict([cleaned_text])[0]
    return prediction

if __name__ == "__main__":
    # If run as a script, try to train on the sample dataset
    data_file = os.path.join("data/tweets.csv")
    if os.path.exists(data_file):
        train_and_save_model(data_file)
    else:
        print(f"No data found at {data_file}. Please place your data there and try again.")
