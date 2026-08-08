import joblib
import pandas as pd
from preprocessing import clean_tweet

def debug_prediction(text):
    model = joblib.load("model.joblib")
    cleaned = clean_tweet(text)
    
    print(f"Original Text: {text}")
    print(f"Cleaned Text: {cleaned}")
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
        classes = model.classes_
        print("\nProbabilities:")
        for cls, prob in zip(classes, probs):
            label = "Positive" if cls == 4 else "Negative" if cls == 0 else str(cls)
            print(f"{label} ({cls}): {prob:.4f}")
    
    prediction = model.predict([cleaned])[0]
    label = "Positive" if prediction == 4 else "Negative" if prediction == 0 else str(prediction)
    print(f"\nFinal Prediction: {label}")

    # Inspect the TF-IDF feature importance if possible
    try:
        tfidf = model.named_steps['tfidf']
        clf = model.named_steps['clf']
        
        feature_names = tfidf.get_feature_names_out()
        coef = clf.coef_[0]
        
        # Get indices of words in the text
        words = cleaned.split()
        print("\nWord Importance (positive values favor Positive sentiment):")
        for word in words:
            if word in tfidf.vocabulary_:
                idx = tfidf.vocabulary_[word]
                importance = coef[idx]
                print(f"'{word}': {importance:.4f}")
            else:
                print(f"'{word}': Not in vocabulary")
    except Exception as e:
        print(f"Could not inspect features: {e}")

if __name__ == "__main__":
    debug_prediction("I feel good today.")
