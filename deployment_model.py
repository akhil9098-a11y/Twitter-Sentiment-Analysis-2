
import joblib

from preprocessing import clean_tweet

MODEL_PATH = "model_tweeteval.joblib"

print("Loading lightweight sentiment model...")

model = joblib.load(MODEL_PATH)

print("Lightweight sentiment model loaded successfully!")

LABEL_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


def predict_sentiment(text):

    cleaned_text = clean_tweet(text)

    if not cleaned_text:
        raise ValueError("Text cannot be empty.")

    probabilities_array = model.predict_proba([cleaned_text])[0]

    probabilities = {
        "Negative": 0.0,
        "Neutral": 0.0,
        "Positive": 0.0
    }

    for label, probability in zip(model.classes_, probabilities_array):
        probabilities[LABEL_MAPPING[int(label)]] = float(probability)

    sentiment = max(probabilities, key=probabilities.get)
    confidence = probabilities[sentiment]

    explanation = (
        f"The model detected predominantly {sentiment.lower()} "
        f"language and assigned {confidence:.1%} confidence "
        f"to the {sentiment} sentiment."
    )

    return {
        "sentiment": sentiment,
        "probabilities": probabilities,
        "explanation": explanation
    }
