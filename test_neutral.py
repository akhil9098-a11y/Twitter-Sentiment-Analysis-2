import joblib
import pandas as pd
from preprocessing import clean_tweet

def test_neutral_logic(text):
    model = joblib.load("model.joblib")
    cleaned_text = clean_tweet(text)
    
    probs = model.predict_proba([cleaned_text])[0]
    classes = model.classes_
    prob_dict = {cls: prob for cls, prob in zip(classes, probs)}
    
    pos_prob = prob_dict.get(4, 0)
    neg_prob = prob_dict.get(0, 0)
    
    sentiment_label = "Neutral"
    if pos_prob > 0.60:
        sentiment_label = "Positive"
    elif neg_prob > 0.60:
        sentiment_label = "Negative"
    
    print(f"Text: {text}")
    print(f"Probabilities: Positive: {pos_prob:.4f}, Negative: {neg_prob:.4f}")
    print(f"Result: {sentiment_label}")

if __name__ == "__main__":
    print("Testing 'I feel good today.'")
    test_neutral_logic("I feel good today.")
    print("\nTesting 'It is a sunny day.' (Should be Neutral-ish)")
    test_neutral_logic("It is a sunny day.")
    print("\nTesting 'I am very happy!' (Should be Positive)")
    test_neutral_logic("I am very happy!")
    print("\nTesting 'I am so sad.' (Should be Negative)")
    test_neutral_logic("I am so sad.")
