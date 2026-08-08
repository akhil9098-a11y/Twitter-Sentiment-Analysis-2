import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from model import load_model
from preprocessing import clean_tweet

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'sans-serif']

# Setup directory for saving images
OUTPUT_DIR = 'static/insights'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling setup
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#0f172a", "figure.facecolor": "#0f172a", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

def generate_visualizations():
    print("Loading a sample of data for visualizations...")
    data_path = "data/tweets.csv"
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Cannot generate visualizations.")
        return

    # Load a sample for speed
    df = pd.read_csv(data_path, encoding='latin-1', header=None, names=['sentiment', 'id', 'date', 'flag', 'user', 'text'])
    df = df.sample(n=min(50000, len(df)), random_state=42)

    # 1. Class Distribution
    print("Generating Class Distribution...")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='sentiment', data=df, palette=['#ef4444', '#34d399'])
    plt.title('Sentiment Class Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sentiment (0 = Negative, 4 = Positive)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=11, color='white', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=150, transparent=True)
    plt.close()

    # Model Evaluation metrics
    try:
        print("Loading Model for Evaluation...")
        model = load_model()
        
        # Clean text first
        print("Cleaning text for predictions...")
        df['cleaned_text'] = df['text'].apply(lambda x: clean_tweet(x) if isinstance(x, str) else "")
        # Drop empty strings after cleaning
        eval_df = df[df['cleaned_text'].str.strip() != '']
        
        y_true = eval_df['sentiment']
        y_pred = model.predict(eval_df['cleaned_text'])

        # 2. Confusion Matrix
        print("Generating Confusion Matrix...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                    annot_kws={"size": 16, "weight": "bold"})
        plt.title('Model Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, transparent=True)
        plt.close()

    except Exception as e:
        print(f"Could not generate model evaluation plots. Error: {e}")

    # 3. Word Clouds
    print("Generating Word Clouds...")
    positive_tweets = " ".join(df[df['sentiment'] == 4]['text'].astype(str))
    negative_tweets = " ".join(df[df['sentiment'] == 0]['text'].astype(str))

    # Positive Wordcloud
    plt.figure(figsize=(10, 5))
    wordcloud_pos = WordCloud(width=800, height=400, background_color='#0f172a', colormap='Greens', max_words=100).generate(positive_tweets)
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words: Positive Tweets', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wordcloud_positive.png'), dpi=150, transparent=True)
    plt.close()

    # Negative Wordcloud
    plt.figure(figsize=(10, 5))
    wordcloud_neg = WordCloud(width=800, height=400, background_color='#0f172a', colormap='Reds', max_words=100).generate(negative_tweets)
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words: Negative Tweets', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wordcloud_negative.png'), dpi=150, transparent=True)
    plt.close()

    print("All visualizations generated successfully and saved in static/insights/!")

if __name__ == "__main__":
    generate_visualizations()
