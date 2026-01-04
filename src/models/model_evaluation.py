import os
import pandas as pd
import pickle
import logging
import yaml
import json
from xgboost import XGBClassifier
import mlflow
from sklearn.metrics import classification_report,accuracy_score,f1_score,recall_score,precision_score

import dagshub
#dagshub.init(repo_owner='Rocky0412', repo_name='Sentiment_Analysis_of_Social_Media_Comments', mlflow=True)

# -------------------- Paths ------------------------
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
input_path = os.path.join(ROOT_DIR, "data", "processed", "test")
model_path = os.path.join(ROOT_DIR, "model")
vectorizer_path=os.path.join(ROOT_DIR,"vectorizer")

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

#----------------------model Evaluation---------------------

def save_model_info(model_info,file_path) -> None:
    try:
       with open(file_path,'w') as f:
           json.dump(model_info,f)

    except Exception as e:
        logger.error(f"Error in save_model_info: {e}")


def model_evaluation(model_path: str, vectorizer_path: str, input_path: str):
    mlflow_uri = "http://ec2-13-232-51-26.ap-south-1.compute.amazonaws.com:8000"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_registry_uri(mlflow_uri)

    mlflow.set_experiment("Sentiment Analysis Using RF")

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Load model + vectorizer
        model = pickle.load(open(os.path.join(model_path, "model.pkl"), "rb"))
        vectorizer = pickle.load(open(os.path.join(vectorizer_path, "vectorizer.pkl"), "rb"))

        xtest = pd.read_csv(os.path.join(input_path, "xtest.csv"))
        ytest = pd.read_csv(os.path.join(input_path, "ytest.csv")).values.ravel()
        ytest = pd.Series(ytest).map({-1:0, 0:1, 1:2}).values

        X_test_text = xtest['clean_comment'].fillna("")
        X_test_transformed = vectorizer.transform(X_test_text)
        ypred = model.predict(X_test_transformed)

        # Metrics
        metric = {
            "accuracy": accuracy_score(ytest, ypred),
            "f1": f1_score(ytest, ypred, average="weighted"),
            "recall": recall_score(ytest, ypred, average="weighted"),
            "precision": precision_score(ytest, ypred, average="weighted"),
        }

        # Save YAML metrics
        metric_dir = os.path.join(ROOT_DIR, "Metric")
        os.makedirs(metric_dir, exist_ok=True)
        yaml_file = os.path.join(metric_dir, "metric.yaml")
        yaml.dump(metric, open(yaml_file, "w"))

        # MLflow logs
        mlflow.set_tags({
            "owner": "Rocky",
            "model": "Random Forest",
            "use_case": "Sentiment Analysis"
        })

        mlflow.log_metrics(metric)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(os.path.join(vectorizer_path, "vectorizer.pkl"))
        mlflow.log_artifact(yaml_file)

        # Build model_info
        model_info = {
            "run_id": run_id,
            "artifact_uri": mlflow.get_artifact_uri(),
            "model_uri": f"runs:/{run_id}/model"
        }

        # Save experiment json
        report_path = os.path.join(ROOT_DIR, "Model_info")
        os.makedirs(report_path, exist_ok=True)
        save_model_info(model_info, os.path.join(report_path, "experiment.json"))

    return metric


            


if __name__ == "__main__":
    model_evaluation(model_path,vectorizer_path,input_path)
