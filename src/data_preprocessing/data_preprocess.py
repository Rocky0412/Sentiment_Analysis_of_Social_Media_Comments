import os
import re
import string
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import yaml 

#--------------------Load params yaml------------------------

# Directory where THIS file lives: src/data
CURRENT_DIR = os.path.dirname(__file__)

# Two levels up → project root (sentiment_analysis/)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

# Full path to params.yaml
PARAMS_PATH = os.path.join(ROOT_DIR, "params.yaml")

with open(PARAMS_PATH,'r') as f:
    params=yaml.safe_load(f)

# ----------------------- LOGGER SETUP -----------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler("logs.log")
    ch = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

# ----------------------- NLTK Downloads -----------------------
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------- Preprocessing Functions -----------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def replace_url(text):
    return re.sub(r'https?:\/\/\S+|www\.\S+', 'URL', text)

def remove_html(text):
    return re.sub(r'<.*?>', '', text)

def replace_mentions(text):
    return re.sub(r'@\S+', 'user', text, flags=re.IGNORECASE)

def replace_num(text):
    return re.sub(r'\d+', 'NUMBER', text)

def replace_heart(text):
    return re.sub(r'<3', 'HEART', text)

def remove_alphanumeric(text):
    return re.sub(r'\w*\d+\w*', '', text)

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in stop_words])

def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def lemmatization(text):
    return ' '.join([lemmatizer.lemmatize(w, pos='v') for w in text.split()])

def clean_text(text):
    text = str(text).lower()
    text = replace_url(text)
    text = remove_html(text)
    text = replace_mentions(text)
    text = replace_num(text)
    text = replace_heart(text)
    text = remove_alphanumeric(text)
    text = remove_stopwords(text)
    text = remove_punctuations(text)
    text = lemmatization(text)
    return text

# ----------------------- Main Pipeline -----------------------
if __name__ == "__main__":
    try:
        logger.info("Reading raw data")
        df = pd.read_csv('data/raw/raw_data.csv')

        if 'clean_comment' not in df.columns or 'category' not in df.columns:
            raise KeyError("Columns 'clean_comment' or 'category' not found in the dataset")

        logger.info("Cleaning text data")
        df['clean_comment'] = df['clean_comment'].apply(clean_text)

        X = df['clean_comment']
        y = df['category']

        # Train-test split
        test_size=params['data_preprocessing']['test_size']
        logger.info("Splitting dataset")
        xtrain, xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Save processed data
        logger.info("Saving processed data")
        os.makedirs('data/processed/train', exist_ok=True)
        xtrain.to_csv('data/processed/train/xtrain.csv', index=False)
        ytrain.to_csv('data/processed/train/ytrain.csv', index=False)
        os.makedirs('data/processed/test', exist_ok=True)
        xtest.to_csv('data/processed/test/xtest.csv', index=False)
        ytest.to_csv('data/processed/test/ytest.csv', index=False)

        logger.info("Data Preprocessing completed successfully")

    except Exception as e:
        logger.exception("Error in data preprocessing pipeline: %s", e)
