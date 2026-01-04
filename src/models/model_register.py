import os
import pandas as pd
import logging
import yaml
import json
import mlflow
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import dagshub
#dagshub.init(repo_owner='Rocky0412', repo_name='Sentiment_Analysis_of_Social_Media_Comments', mlflow=True)
# -------------------- Paths ------------------------
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
input_path = os.path.join(ROOT_DIR, "data", "processed", "test")
model_path = os.path.join(ROOT_DIR, "model")
vectorizer_path = os.path.join(ROOT_DIR, "vectorizer")

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


# ---------------------- Load Model Info ----------------------------
def load_model_info(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            data = json.load(f)

        logger.info('Model info loaded')
        return data

    except Exception as e:
        logger.error(f"Error loading the JSON file: {e}")
        return {}


# ---------------------- Register Model ----------------------------
'''def register_model(model_info):
    try:
        run_id = model_info["run_id"]
        artifact_path = model_info.get("artifact_path", model_info.get("model", "model"))
        print(f'model path : {artifact_path}')

        # correct f-string
        model_uri = f"runs:/{run_id}/{artifact_path}"

        logger.info(f"Registering model from URI: {model_uri}")

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name="sentiment_analysis_model"
        )

        client=mlflow.client.MlflowClient()
        client.transition_model_version_stage(
            name='sentiment_analysis_model',
            version=model_version.version,
            stage="Staging"
        )

        logger.info(f"Model registered successfully: version={model_version.version}")

    except Exception as e:
        logger.error(f"Model registration failed: {e}")

'''
mlflow_uri='http://ec2-13-232-51-26.ap-south-1.compute.amazonaws.com:8000'
#mlflow.set_registry_uri(uri='https://dagshub.com/Rocky0412/Sentiment_Analysis_of_Social_Media_Comments.mlflow')
mlflow.set_tracking_uri(mlflow_uri) #used to set uri
mlflow.set_registry_uri(uri=mlflow_uri) #used to register model

def register_model(model_info):
    try:
        # Prefer using the full artifact_uri if available
        if "artifact_uri" in model_info:
            model_uri = model_info["artifact_uri"]
        else:
            # fallback to run_id + artifact path
            run_id = model_info["run_id"]
            artifact_path = model_info.get("model", "model")
            model_uri = f"runs:/{run_id}/{artifact_path}"

        logger.info(f"Registering model from URI: {model_uri}")

        client = mlflow.tracking.MlflowClient()
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name="sentiment_analysis_model"
        )

        client.transition_model_version_stage(
            name='sentiment_analysis_model',
            version=model_version.version,
            stage="Staging"
        )

        logger.info(f"Model registered successfully: version={model_version.version}")

    except Exception as e:
        logger.error(f"Model registration failed: {e}")


# --------------------- MAIN -------------------------
if __name__ == '__main__':
    model_info_path = os.path.join(ROOT_DIR, 'model_info', 'experiment.json')
    model_info = load_model_info(model_info_path)

    register_model(model_info)

