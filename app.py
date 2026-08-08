import streamlit as st
import os
from model import load_model, predict_sentiment, train_and_save_model

# Streamlit UI Setup
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        text-align: center;
        background: -webkit-linear-gradient(#1DA1F2, #0d8bd9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #8899a6;
        margin-bottom: 2rem;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .sentiment-positive { background-color: rgba(46, 204, 113, 0.2); border: 1px solid #2ecc71; color: #2ecc71; }
    .sentiment-negative { background-color: rgba(231, 76, 60, 0.2); border: 1px solid #e74c3c; color: #e74c3c; }
    .sentiment-neutral { background-color: rgba(241, 196, 15, 0.2); border: 1px solid #f1c40f; color: #f1c40f; }
    .sentiment-unknown { background-color: rgba(149, 165, 166, 0.2); border: 1px solid #95a5a6; color: #95a5a6; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Twitter Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Analyze the emotion behind any tweet or text snippet.</p>", unsafe_allow_html=True)

# Application logic
@st.cache_resource
def get_model():
    model_path = "model.joblib"
    if not os.path.exists(model_path):
        # Automatically train on sample data if model doesn't exist
        sample_data = "data/sample_tweets.csv"
        if os.path.exists(sample_data):
            st.info("Model not found. Training on sample dataset automatically...")
            train_and_save_model(sample_data)
        else:
            return None
    try:
        return load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model()

if model is None:
    st.error("Model could not be loaded. Please ensure 'model.joblib' exists or 'data/sample_tweets.csv' is present to train a new one.")
else:
    # Input Area
    user_input = st.text_area("Enter a tweet or text to analyze:", height=150, placeholder="E.g., I absolutely love the new design!")

    if st.button("Analyze Sentiment", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                prediction = predict_sentiment(user_input, model)
                
                # Display results with style
                sentiment_class = prediction.lower()
                
                if 'positive' in sentiment_class:
                    css_class = 'sentiment-positive'
                    emoji = '😃'
                elif 'negative' in sentiment_class:
                    css_class = 'sentiment-negative'
                    emoji = '😠'
                elif 'neutral' in sentiment_class:
                    css_class = 'sentiment-neutral'
                    emoji = '😐'
                else:
                    css_class = 'sentiment-unknown'
                    emoji = '🤔'
                
                st.markdown(f"""
                <div class='sentiment-box {css_class}'>
                    {emoji} Predicted Sentiment: {prediction.capitalize()}
                </div>
                """, unsafe_allow_html=True)
                
                # Show confidence or additional details if needed (optional)
                # If using predict_proba, we could show probabilities here.
                if hasattr(model, "predict_proba"):
                    from preprocessing import clean_tweet
                    cleaned = clean_tweet(user_input)
                    if cleaned:
                        probs = model.predict_proba([cleaned])[0]
                        classes = model.classes_
                        st.write("---")
                        st.write("Confidence Breakdown:")
                        for cls, prob in zip(classes, probs):
                            st.progress(float(prob), text=f"{cls.capitalize()}: {prob:.1%}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #8899a6; font-size: 0.8rem;'>Powered by Scikit-Learn & Streamlit</div>", unsafe_allow_html=True)
