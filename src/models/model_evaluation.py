import os
import pandas as pd
import logging
import yaml
import json
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

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

# ---------------------- MODEL EVALUATION ---------------------
def save_model_info(model_info, file_path) -> None:
    try:
        with open(file_path, "w") as f:
            json.dump(model_info, f)
    except Exception as e:
        logger.error(f"Error saving model info: {e}")

def model_evaluation(model_path: str, vectorizer_path: str, input_path: str):
    # MLflow setup
    mlflow_uri = "http://ec2-13-232-51-26.ap-south-1.compute.amazonaws.com:8000"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_registry_uri(mlflow_uri)
    mlflow.set_experiment("Sentiment Analysis Using RF")

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # ---------------- Load model + vectorizer ----------------
        import pickle
        with open(os.path.join(model_path, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(vectorizer_path, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

        # ---------------- Load test data ----------------
        xtest = pd.read_csv(os.path.join(input_path, "xtest.csv"))
        ytest = pd.read_csv(os.path.join(input_path, "ytest.csv")).values.ravel()
        ytest = pd.Series(ytest).map({-1:0, 0:1, 1:2}).values
        X_test_text = xtest['clean_comment'].fillna("")
        X_test_transformed = vectorizer.transform(X_test_text)
        ypred = model.predict(X_test_transformed)

        # ---------------- Metrics ----------------
        metric = {
<<<<<<< HEAD
            "accuracy": accuracy_score(ytest, ypred),
            "f1": f1_score(ytest, ypred, average="weighted"),
            "recall": recall_score(ytest, ypred, average="weighted"),
            "precision": precision_score(ytest, ypred, average="weighted")
=======
                "accuracy": acc,
                "f1": f1,
                "recall": recall,
                "precision": precision,
            }

            # File path for YAML
        with open(os.path.join(metric_dir,'metric.yaml'),'w') as f:
            yaml.dump(metric,f,indent=4)

        #mlflow code
        mlflow.set_tags({
        "owner": "Rocky",
        "use_case": "sentiment_analysis",
        "env": "local",
        "model":'Random Forest'
        })
            
        mlflow.log_metrics(
            metrics=metric
        )
        logged_model=mlflow.sklearn.log_model(model,"model")

        mlflow.log_artifact(vectorizer_path)

        artifact_uri='model'

        model_info={
            "run_id" :run_id,
            "artifact_uri": artifact_uri,
            "model":"model",
            "model_uri":logged_model.model_uri
>>>>>>> eb0dce0
        }

        # Save metrics to YAML
        metric_dir = os.path.join(ROOT_DIR, "Metric")
        os.makedirs(metric_dir, exist_ok=True)
        yaml_file = os.path.join(metric_dir, "metric.yaml")
        yaml.dump(metric, open(yaml_file, "w"))

        # ---------------- MLflow logging ----------------
        mlflow.set_tags({
            "owner": "Rocky",
            "model": "Random Forest",
            "use_case": "Sentiment Analysis"
        })
        mlflow.log_metrics(metric)

        # ✅ Properly log sklearn model (creates MLmodel automatically)
        logged_model = mlflow.sklearn.log_model(model, artifact_path="model")

        # Log vectorizer and metrics artifacts
        mlflow.log_artifact(os.path.join(vectorizer_path, "vectorizer.pkl"))
        mlflow.log_artifact(yaml_file)

        # ---------------- Model info ----------------
        model_info = {
            "run_id": run_id,
            "artifact_uri": mlflow.get_artifact_uri(),
            "model_uri": logged_model.model_uri
        }

        # Save experiment json
        report_path = os.path.join(ROOT_DIR, "Model_info")
        os.makedirs(report_path, exist_ok=True)
        save_model_info(model_info, os.path.join(report_path, "experiment.json"))

        # ---------------- Optional: Register model ----------------
        try:
            mlflow.register_model(model_uri=logged_model.model_uri, name="sentiment_analysis_model")
        except Exception as e:
            logger.warning(f"Model registration skipped: {e}")

    return metric

if __name__ == "__main__":
    model_evaluation(model_path, vectorizer_path, input_path)
