import os
import pandas as pd
import logging
import yaml
import json
import mlflow
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

# -------------------- Paths ------------------------
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
model_info_path = os.path.join(ROOT_DIR, 'model_info', 'experiment.json')

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

        logger.info('Model info loaded successfully')
        return data

    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        return {}

# ---------------------- MLflow URIs ----------------------------
mlflow_uri = 'http://ec2-13-232-51-26.ap-south-1.compute.amazonaws.com:8000'

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)


# ---------------------- Register Model ----------------------------
<<<<<<< HEAD
=======
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

# ---------------------- Register Model ----------------------------
>>>>>>> eb0dce0
def register_model(model_info):
    try:
        client = mlflow.tracking.MlflowClient()

<<<<<<< HEAD
        # CASE 1: If artifact_uri exists
        if "artifact_uri" in model_info:
            base_uri = model_info["artifact_uri"]

            # Replace run-specific artifact dir with "model"
            if base_uri.endswith(model_info["run_id"]):
                base_uri = base_uri.rsplit("/", 1)[0]  # remove run_id

            model_uri = f"{base_uri}/model"

        # CASE 2: fallback using run_id + artifact path
        else:
            run_id = model_info["run_id"]
            model_uri = f"runs:/{run_id}/model"

        logger.info(f"Registering model with URI: {model_uri}")
=======
        # Use run_id + artifact folder instead of artifact_uri
        run_id = model_info.get("run_id")
        if not run_id:
            raise ValueError("run_id is missing in model_info!")

        artifact_path = model_info.get("model", "model")  # usually "model"
        model_uri = f"runs:/{run_id}/{artifact_path}"  # correct MLflow URI

        model_name = "my_model"  # keep a consistent name
        logger.info(f"Registering model from URI: {model_uri}")
>>>>>>> eb0dce0

        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
<<<<<<< HEAD
        logger.info(f"Model registered successfully as version {model_version.version}")
        # Move model to staging
        client.transition_model_version_stage(
            name="sentiment_analysis_model",
=======

        # Move model to Staging
        client.transition_model_version_stage(
            name=model_name,
>>>>>>> eb0dce0
            version=model_version.version,
            stage="Staging"
        )

        logger.info(f"Model registered successfully as version {model_version.version}")

    except Exception as e:
        logger.error(f"Model registration failed: {e}")

# --------------------- MAIN -------------------------
if __name__ == '__main__':
    model_info = load_model_info(model_info_path)
    register_model(model_info)


