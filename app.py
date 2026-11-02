# from evaluate_roberta import result
import os

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

from preprocessing import clean_tweet
from deployment_model import predict_sentiment


app = Flask(__name__)

# Database location can be configured through an environment variable.
database_path = os.environ.get(
    "DATABASE_PATH",
    os.path.join(app.instance_path, "sentiment.db")
)

# Ensure the database directory exists.
os.makedirs(os.path.dirname(database_path), exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + database_path.replace("\\", "/")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class AnalysisHistory(db.Model):
    __tablename__ = "analysis_history"

    id = db.Column(db.Integer, primary_key=True)

    text = db.Column(db.Text, nullable=False)

    sentiment = db.Column(db.String(20), nullable=False)

    negative_probability = db.Column(db.Float, nullable=False)
    neutral_probability = db.Column(db.Float, nullable=False)
    positive_probability = db.Column(db.Float, nullable=False)

    explanation = db.Column(db.Text, nullable=True)

    created_at = db.Column(
        db.DateTime,
        default=db.func.current_timestamp()
    )

    # Initialize database tables when the application starts.
with app.app_context():
    db.create_all()
    print("Database tables initialized successfully!", flush=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({
                "error": "No text provided."
            }), 400

        text = data["text"].strip()

        if not text:
            return jsonify({
                "error": "Please enter some text to analyze."
            }), 400

        cleaned_text = clean_tweet(text)

        if not cleaned_text:
            return jsonify({
                "error": "Text became empty after preprocessing."
            }), 400

        result = predict_sentiment(cleaned_text)

# Token-level RoBERTa explainability is disabled
# in the lightweight deployment configuration.
        result["token_importance"] = []

        history = AnalysisHistory(
            text=text,
            sentiment=result["sentiment"],
            negative_probability=result["probabilities"]["Negative"],
            neutral_probability=result["probabilities"]["Neutral"],
            positive_probability=result["probabilities"]["Positive"],
            explanation=result["explanation"]
        )

        db.session.add(history)
        db.session.commit()


        return jsonify(result)

    except Exception as e:

        print(f"Prediction error: {e}")

        return jsonify({
            "error": "An error occurred while analyzing the text."
        }), 500

@app.route("/api/history", methods=["GET"])
def get_history():

    try:
        records = AnalysisHistory.query.order_by(
            AnalysisHistory.created_at.desc()
        ).all()

        history = []

        for record in records:

            history.append({
                "id": record.id,
                "text": record.text,
                "sentiment": record.sentiment,
                "probabilities": {
                    "Negative": record.negative_probability,
                    "Neutral": record.neutral_probability,
                    "Positive": record.positive_probability
                },
                "explanation": record.explanation,
                "created_at": record.created_at.isoformat()
            })

        return jsonify({
            "total": len(history),
            "history": history
        })

    except Exception as e:

        print(f"History retrieval error: {e}")

        return jsonify({
            "error": "Unable to retrieve analysis history."
        }), 500

@app.route("/api/history/<int:history_id>", methods=["DELETE"])
def delete_history(history_id):

    try:
        record = db.session.get(AnalysisHistory, history_id)

        if record is None:
            return jsonify({
                "error": "History record not found."
            }), 404

        db.session.delete(record)
        db.session.commit()

        return jsonify({
            "message": "History record deleted successfully.",
            "id": history_id
        })

    except Exception as e:
        db.session.rollback()
        print(f"History deletion error: {e}")

        return jsonify({
            "error": "Unable to delete history record."
        }), 500

@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():

    try:
        data = request.get_json()

        if not data or "texts" not in data:
            return jsonify({
                "error": "No texts provided."
            }), 400

        texts = data["texts"]

        if not isinstance(texts, list):
            return jsonify({
                "error": "Texts must be provided as a list."
            }), 400

        if len(texts) == 0:
            return jsonify({
                "error": "Please provide at least one text."
            }), 400

        if len(texts) > 50:
            return jsonify({
                "error": "Maximum 50 texts can be analyzed at once."
            }), 400

        
        results = []

        for text in texts:

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            cleaned_text = clean_tweet(text)

            if not cleaned_text:
                continue

            result = predict_sentiment(cleaned_text)

            # Save each batch prediction to analysis history
            history = AnalysisHistory(
                text=text,
                sentiment=result["sentiment"],
                negative_probability=result["probabilities"]["Negative"],
                neutral_probability=result["probabilities"]["Neutral"],
                positive_probability=result["probabilities"]["Positive"],
                explanation=result["explanation"]
            )

            db.session.add(history)

            # Append each prediction to the results list
            results.append({
                "text": text,
                "sentiment": result["sentiment"],
                "probabilities": result["probabilities"]
            })


        if not results:
            return jsonify({
                "error": "No valid texts were provided."
            }), 400

        sentiment_counts = {
            "Negative": 0,
            "Neutral": 0,
            "Positive": 0
        }

        for result in results:
            sentiment_counts[result["sentiment"]] += 1

        total = len(results)

        sentiment_percentages = {
            sentiment: round((count / total) * 100, 2)
            for sentiment, count in sentiment_counts.items()
        }

        # Calculate average prediction confidence
        total_confidence = sum(
            max(result["probabilities"].values())
            for result in results
        )

        average_confidence = round(
            (total_confidence / total) * 100, 2
        )


        # Find the most positive tweet
        positive_results = [
            result for result in results
            if result["sentiment"] == "Positive"
        ]

        most_positive = None

        if positive_results:
            most_positive = max(
            positive_results,
            key=lambda x: x["probabilities"]["Positive"]
    )


# Find the most negative tweet
        negative_results = [
        result for result in results
        if result["sentiment"] == "Negative"
        ]

        most_negative = None

        if negative_results:
            most_negative = max(
            negative_results,
            key=lambda x: x["probabilities"]["Negative"]
            )


        # Prepare strongest sentiment examples
        strongest_examples = {
            "most_positive": most_positive,
            "most_negative": most_negative
        }
        
        # Save all batch predictions to database
        db.session.commit()

        return jsonify({
            "total": total,
            "results": results,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "average_confidence": average_confidence,
            "strongest_examples": strongest_examples
        })

    except Exception as e:

        db.session.rollback()
        print(f"Batch prediction error: {e}")

        return jsonify({
            "error": "An error occurred while analyzing the texts."
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
