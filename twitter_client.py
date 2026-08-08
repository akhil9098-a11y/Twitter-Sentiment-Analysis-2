import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

class TwitterClient:
    def __init__(self):
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        self.client = None
        if self.bearer_token:
            try:
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_token_secret
                )
            except Exception as e:
                print(f"Error initializing Tweepy Client: {e}")

    def fetch_tweets(self, query, max_results=100):
        """
        Fetches recent tweets based on a query.
        Note: Free tier has limitations on search.
        """
        if not self.client:
            print("Twitter Client not initialized. Check your .env file.")
            return []

        try:
            # Search for recent tweets
            # 'lang:en' filters for English, '-is:retweet' excludes retweets
            full_query = f"{query} lang:en -is:retweet"
            response = self.client.search_recent_tweets(query=full_query, max_results=max_results)
            
            if not response.data:
                return []
            
            return [tweet.text for tweet in response.data]
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return []

    def get_labeled_data(self, limit_per_sentiment=100):
        """
        Bootstraps labeled data using emojis as sentiment markers.
        Positive: :) , :-D , :-)
        Negative: :( , :-(
        """
        print(f"Fetching labeled data from Twitter (limit: {limit_per_sentiment} per sentiment)...")
        
        # Positive query
        pos_query = "(:) OR :-D OR :-))"
        pos_tweets = self.fetch_tweets(pos_query, max_results=limit_per_sentiment)
        
        # Negative query
        neg_query = "(:( OR :-( OR :(("
        neg_tweets = self.fetch_tweets(neg_query, max_results=limit_per_sentiment)
        
        data = []
        for text in pos_tweets:
            data.append({'text': text, 'sentiment': 4}) # 4 = Positive in Sentiment140
            
        for text in neg_tweets:
            data.append({'text': text, 'sentiment': 0}) # 0 = Negative in Sentiment140
            
        return data

if __name__ == "__main__":
    # Test client
    client = TwitterClient()
    if client.client:
        print("Twitter Client connected successfully.")
        # test_tweets = client.fetch_tweets("Python", max_results=10)
        # print(f"Found {len(test_tweets)} tweets.")
    else:
        print("Failed to connect. Ensure .env is set up correctly.")
