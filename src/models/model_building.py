import os
import pandas as pd
import pickle
import logging
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------- Paths ------------------------
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
input_path = os.path.join(ROOT_DIR, "data", "processed", "train")
model_path = os.path.join(ROOT_DIR, "model")

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

def train_model(input_path, model_path):

    logger.info("Loading training data...")
    X = pd.read_csv(os.path.join(input_path, "xtrain_vectorized.csv"))
    y = pd.read_csv(os.path.join(input_path, "ytrain_vectorized.csv")).values.ravel()

    # ------------------------------------------------------
    # FIX LABELS: Convert [-1, 0, 1] → [0, 1, 2] (Required by XGBoost)
    # ------------------------------------------------------
    logger.info("Converting labels -1,0,1 → 0,1,2 for XGBoost compatibility...")
    y = pd.Series(y).map({-1: 0, 0: 1, 1: 2}).values

    logger.info("Initializing XGBoost classifier for multiclass...")


    model=RandomForestClassifier(max_depth=15,n_estimators=100,criterion='entropy')

    logger.info("Training RF model...")
    model.fit(X, y)

    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    logger.info("XGBoost model trained and saved ✔")
    return model


if __name__ == "__main__":
    train_model(input_path, model_path)
