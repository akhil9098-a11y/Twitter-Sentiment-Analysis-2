import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required NLTK resources quietly
def download_nltk_resources():
    resources = [
        'stopwords',
        'wordnet',
        'omw-1.4',
        'punkt',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]
    for resource in resources:
        try:
            if resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'wordnet':
                nltk.data.find('corpora/wordnet')
            elif resource == 'omw-1.4':
                nltk.data.find('corpora/omw-1.4')
            elif resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif resource == 'averaged_perceptron_tagger':
                nltk.data.find('taggers/averaged_perceptron_tagger')
            elif resource == 'averaged_perceptron_tagger_eng':
                nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_resources()

# Refine stopword list to keep sentiment-relevant words
# We want to keep words that change the meaning of the sentence significantly
sentiment_exceptions = {
    'not', 'no', 'never', 'nor', 'neither', 'but', 'however', 'none', 
    'nothing', 'nowhere', 'nobody', 'none', 'cannot', 'cant', 'couldnt', 
    'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt', 'havent', 'isnt', 
    'mightnt', 'mustnt', 'neednt', 'shant', 'shouldnt', 'wasnt', 
    'werent', 'wont', 'wouldnt'
}
stop_words = set(stopwords.words('english')) - sentiment_exceptions
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Map POS tag to first character lemmatize() accepts"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_tweet(tweet):
    """
    Cleans a single tweet string using proper tokenization, 
    POS-aware lemmatization, and refined stopword removal.
    """
    if not isinstance(tweet, str):
        return ""
    
    # Lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags (keeping the hashtag word itself often helps)
    tweet = re.sub(r'\@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    
    # Tokenize properly
    tokens = nltk.word_tokenize(tweet)
    
    # Remove punctuations and non-alphabetic tokens, then POS tag
    tokens = [word for word in tokens if word.isalnum()]
    
    # POS tagging
    pos_tags = nltk.pos_tag(tokens)
    
    # Lemmatize with POS tags and remove stopwords
    cleaned_tokens = []
    for word, tag in pos_tags:
        if word not in stop_words:
            pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos)
            cleaned_tokens.append(lemma)
    
    return " ".join(cleaned_tokens)

def preprocess_data(df, text_column='text'):
    """
    Preprocesses a pandas DataFrame containing a text column.
    """
    df['cleaned_text'] = df[text_column].apply(clean_tweet)
    return df
