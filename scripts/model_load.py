
import mlflow # type: ignore
from mlflow.tracking import MlflowClient # type: ignore
import pickle
import os

mlflow_uri = 'http://ec2-13-232-51-26.ap-south-1.compute.amazonaws.com:8000'
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)

# ------------------------------------------------------
# 1️⃣ Load MLflow Model (latest from stage)
# ------------------------------------------------------
def load_model(model_name="sentiment_analysis_model", stage="Staging"):
    """
    Loads the latest model from MLflow model registry by stage.
    Returns a PyFunc model.
    """
    client = MlflowClient()

    # Get newest version from stage
    latest = client.get_latest_versions(model_name, stages=[stage])
    if not latest:
        raise ValueError(f"No model found in stage '{stage}' for '{model_name}'")

    version = latest[0].version
    model_uri = f"models:/{model_name}/{stage}"

    print(f"Loading model '{model_name}' (version {version}) from stage: {stage}")

    return mlflow.pyfunc.load_model(model_uri)


# ------------------------------------------------------
# 2️⃣ Load vectorizer from run artifacts
# ------------------------------------------------------
def load_vectorizer(run_id: str, artifact_path="vectorizer", file_name="vectorizer.pkl"):
    """
    Downloads the vectorizer artifact folder and loads vectorizer.pkl.
    """
    client = MlflowClient()

    # Download artifact folder locally
    local_path = client.download_artifacts(run_id, artifact_path)

    vectorizer_file = os.path.join(local_path, file_name)

    with open(vectorizer_file, "rb") as f:
        vectorizer = pickle.load(f)

    print("Vectorizer loaded successfully!")
    return vectorizer

# 1️⃣ Load MLflow Model
model = load_model(model_name="my_model", stage="Staging")
# 2️⃣ Load Vectorizer
run_id = "e8fb363096b247d68c92007cd0da948c"
vectorizer = load_vectorizer(run_id)

if __name__ == "__main__":


    # Test Input
    text =['I have Mangoes']

    X = vectorizer.transform(text)
    prediction = model.predict(X)

    print("Prediction:", prediction)

