import pandas as pd
import os
from model import train_and_save_model

# Create a small sample of the data for quick training
base_data = "data/tweets.csv"
sample_data = "data/sample_tweets.csv"

if os.path.exists(base_data):
    print("Reading base data for sampling...")
    df = pd.read_csv(base_data, encoding='latin-1', header=None, names=['sentiment', 'id', 'date', 'flag', 'user', 'text'], nrows=1000)
    df.to_csv(sample_data, index=False)
    print(f"Sample data saved to {sample_data}")
    
    print("Training model on sample data...")
    train_and_save_model(sample_data)
else:
    print(f"Base data not found at {base_data}")
