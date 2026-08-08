from flask import Flask, request, jsonify, render_template
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'sans-serif']
from model import load_model, predict_sentiment
from preprocessing import clean_tweet

app = Flask(__name__)

model = None
try:
    model = load_model()
except Exception as e:
    print(f"Warning: Could not load model initially: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            return jsonify({'error': 'Model is not trained or cannot be loaded. Run model.py first.'}), 500
            
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
        
    text = data['text']
    try:
        prediction = predict_sentiment(text, model)
        
        # Default labels
        sentiment_label = "Neutral"
        
        # Calculate probabilities for threshold-based neutral detection
        cleaned_text = clean_tweet(text)
        if hasattr(model, 'predict_proba') and cleaned_text:
            probs = model.predict_proba([cleaned_text])[0]
            classes = model.classes_
            
            # Map probabilities to labels
            prob_dict = {cls: prob for cls, prob in zip(classes, probs)}
            pos_prob = prob_dict.get(4, 0)
            neg_prob = prob_dict.get(0, 0)
            
            # Threshold: If neither is dominant (> 60%), it's Neutral
            if pos_prob > 0.60:
                sentiment_label = "Positive"
            elif neg_prob > 0.60:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
        else:
            # Fallback to direct prediction if no probabilities available
            if prediction == 0:
                sentiment_label = "Negative"
            elif prediction == 4:
                sentiment_label = "Positive"
            
        # Prepare probabilities for the frontend graph
        probabilities = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
        
        if hasattr(model, 'predict_proba') and cleaned_text:
            probs = model.predict_proba([cleaned_text])[0]
            classes = model.classes_
            
            prob_dict = {cls: prob for cls, prob in zip(classes, probs)}
            pos_p = prob_dict.get(4, 0)
            neg_p = prob_dict.get(0, 0)
            
            # Calculate a pseudo-neutral score for the graph if using a 2-class model
            # Neutrality is high when confidence is low (near 50/50)
            neutral_p = 1 - abs(pos_p - neg_p)
            
            # Normalize to sum to 1 if you want them to behave like probabilities
            # (Though technically these are overlapping concepts)
            total = pos_p + neg_p + neutral_p
            
            probabilities = {
                'Negative': float(neg_p / total),
                'Neutral': float(neutral_p / total),
                'Positive': float(pos_p / total)
            }
            
        return jsonify({
            'sentiment': sentiment_label,
            'original_text': text,
            'probabilities': probabilities
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
