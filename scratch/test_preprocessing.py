from preprocessing import clean_tweet

test_tweets = [
    "I am not happy with this product! #fail",
    "The weather is wonderful today! http://weather.com",
    "I've been running for hours and I'm exhausted.",
    "@user This is a great movie, I loved it!",
    "It's not that bad, actually."
]

print("Original Preprocessing Results:")
for tweet in test_tweets:
    print(f"Original: {tweet}")
    print(f"Cleaned:  {clean_tweet(tweet)}")
    print("-" * 20)
