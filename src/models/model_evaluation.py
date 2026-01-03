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
dagshub.init(repo_owner='Rocky0412', repo_name='Sentiment_Analysis_of_Social_Media_Comments', mlflow=True)

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
    if model_path is None or vectorizer_path is None:
        logger.error("Invalid path")
        raise ValueError("Invalid path")
    
    mlflow.set_registry_uri(uri='https://dagshub.com/Rocky0412/Sentiment_Analysis_of_Social_Media_Comments.mlflow')

    mlflow.set_experiment('Sentiment Analysis Using RF')
    
    with mlflow.start_run() as run:

        run_id=run.info.run_id

        # Load trained model
        with open(os.path.join(model_path, "model.pkl"), "rb") as f:
            model = pickle.load(f)

        # Load vectorizer
        with open(os.path.join(vectorizer_path, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

        # Load test data
        xtest = pd.read_csv(os.path.join(input_path, "xtest.csv"))
        ytest = pd.read_csv(os.path.join(input_path, "ytest.csv")).values.ravel()

        # Map labels -1,0,1 → 0,1,2 (same as training)
        ytest = pd.Series(ytest).map({-1: 0, 0: 1, 1: 2}).values

        logger.info("Transforming test data using vectorizer...")
        X_test_text = xtest['clean_comment'].fillna("") 
        X_test_transformed = vectorizer.transform(X_test_text)

        logger.info("Predicting test data...")
        ypred = model.predict(X_test_transformed)

        # Metrics
        acc = accuracy_score(ytest, ypred)
        f1 = f1_score(ytest, ypred, average="weighted")
        recall = recall_score(ytest, ypred, average="weighted")
        precision = precision_score(ytest, ypred, average="weighted")
        report = classification_report(ytest, ypred)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
                # Assuming ROOT_DIR is defined
        metric_dir = os.path.join(ROOT_DIR, 'Metric')

            # Create the directory if it doesn't exist
        os.makedirs(metric_dir, exist_ok=True)
        print(metric_dir)

        # Metrics dictionary
        metric = {
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
        mlflow.sklearn.log_model(model,"model")

        mlflow.log_artifact(vectorizer_path)

        artifact_uri=mlflow.get_artifact_uri(run_id)

        model_info={
            "run_id" :run_id,
            "artifact_uri": artifact_uri,
            "model":"model"
        }

        '''model_uri = f"runs:/{run_id}/rf_model"
        mlflow.register_model(
            model_uri=model_uri,
            name="sentiment_analysis_model"
        )
        client=mlflow.client.MlflowClient()'''

        report_path=os.path.join(ROOT_DIR,'Model_info')
        os.makedirs(report_path,exist_ok=True)
        file_path=os.path.join(report_path,'experiment.json')
        save_model_info(model_info=model_info,file_path=file_path)

    return metric

            


if __name__ == "__main__":
    model_evaluation(model_path,vectorizer_path,input_path)
