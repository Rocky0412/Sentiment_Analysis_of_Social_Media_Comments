import os
import json
import logging
import mlflow
from mlflow.tracking import MlflowClient

# -------------------- Paths ------------------------
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
MODEL_INFO_PATH = os.path.join(ROOT_DIR, "model_info", "experiment.json")

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

# ---------------------- MLflow Setup ------------------------
MLFLOW_URI = "http://ec2-13-232-51-26.ap-south-1.compute.amazonaws.com:8000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

# ---------------------- Load Model Info ---------------------
def load_model_info(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model info file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if "run_id" not in data:
        raise KeyError("run_id missing in experiment.json")

    logger.info("Model info loaded successfully")
    return data

# ---------------------- Register Model ---------------------
def register_model(model_info: dict):
    try:
        run_id = model_info["run_id"]
        artifact_path = model_info.get("model", "model")  # default = "model"

        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_name = "sentiment_analysis_model"

        logger.info(f"Registering model from URI: {model_uri}")

        client = MlflowClient()

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.info(
            f"Model registered successfully: "
            f"name={model_name}, version={model_version.version}"
        )

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise  # fail fast in CI

# ---------------------- MAIN ------------------------
if __name__ == "__main__":
    model_info = load_model_info(MODEL_INFO_PATH)
    register_model(model_info)
