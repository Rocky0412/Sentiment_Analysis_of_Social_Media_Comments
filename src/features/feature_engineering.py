from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import os
import logging
import yaml
import pickle

# -------------------- Load params yaml ------------------------
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
PARAMS_PATH = os.path.join(ROOT_DIR, "params.yaml")

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

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


def feature_extraction(path: str = None) -> None:
    logger.info("Loading training data")

    xtrain = pd.read_csv(os.path.join(path, "train/xtrain.csv"))
    ytrain = pd.read_csv(os.path.join(path, "train/ytrain.csv"))

    # ---------- FIX: Replace NaN text with empty string ----------
    logger.info("Cleaning NaN values")
    text_data = xtrain.iloc[:, 0].fillna("").astype(str)

    logger.info("Vectorizing using TF-IDF")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
    X = vectorizer.fit_transform(text_data)

    logger.info("Applying SMOTE")
    smote = SMOTE(random_state=42)

    X_dense = X.toarray()  # SMOTE needs dense matrix
    X_res, y_res = smote.fit_resample(X_dense, ytrain)

    logger.info("Saving vectorizer and processed features")

    # Save vectorizer
    vectorizer_dir = os.path.join(ROOT_DIR, "vectorizer")
    os.makedirs(vectorizer_dir, exist_ok=True)

    with open(os.path.join(vectorizer_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save resampled train data
    pd.DataFrame(X_res).to_csv(os.path.join(path, "train/xtrain_vectorized.csv"), index=False)
    y_res.to_csv(os.path.join(path, "train/ytrain_vectorized.csv"), index=False)

    logger.info("Feature extraction completed successfully")

    return vectorizer


if __name__ == "__main__":
    data_processed_path = os.path.join(ROOT_DIR, "data/processed")
    feature_extraction(data_processed_path)






