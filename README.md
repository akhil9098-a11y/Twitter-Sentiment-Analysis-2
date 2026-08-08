# Twitter Sentiment Analysis

A comprehensive Machine Learning web application to predict the sentiment of tweets or text snippets. It categorizes text into positive or negative sentiments using a Scikit-Learn TF-IDF + Logistic Regression pipeline. 

The project features a sleek, premium, and fully-responsive web application built with Flask, HTML, CSS, and JavaScript. A legacy Streamlit interface is also included as a backup.

## Directory Structure
- `data/`: Contains the datasets used for training (e.g., `tweets.csv`).
- `preprocessing.py`: Contains NLP logic for text cleaning, stopword removal, and lemmatization using `nltk`.
- `model.py`: Handles model building, training, evaluation, and saving using `scikit-learn` and `joblib`.
- `web_app.py`: The main Flask server exposing the API and serving the web interface.
- `templates/` & `static/`: Contains the modern HTML structure, vanilla CSS (glassmorphism design), and JS logic.
- `app.py`: The legacy Streamlit web application.
- `requirements.txt`: Python package dependencies.

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed (Python 3.8+ recommended). Open your terminal and run:
```bash
pip install -r requirements.txt
```
*(NLTK stopwords and wordnet corpora will be downloaded automatically when preprocessing runs).*

### 2. Prepare the Model
A pre-trained model `model.joblib` might already be included in the project folder. If it is missing, or if you wish to retrain it on your own dataset, simply place your data inside the `data/` folder and run:
```bash
python model.py
```

### 3. Run the Web Application (Recommended)
To launch the modern Flask web application, run:
```bash
python web_app.py
```
The application will start a local server. Open your web browser and navigate to `http://127.0.0.1:5000/` to use the sentiment analyzer!

### 4. Run the Streamlit Application (Legacy)
If you prefer to run the backup Streamlit app:
```bash
streamlit run app.py
```
This will open in your browser at `http://localhost:8501`.
