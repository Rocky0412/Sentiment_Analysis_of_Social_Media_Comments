from flask import Flask, request, jsonify
import mlflow
from mlflow.pyfunc import load_model
from flask_cors import CORS
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import json
import pickle
import os

app = Flask(__name__)
CORS(app)


# --------------- Load MLflow Model -----------------

mlflow_uri = 'http://ec2-13-232-51-26.ap-south-1.compute.amazonaws.com:8000'
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)

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


model = load_model(model_name="my_model", stage="Staging") # Change to your MLflow path

with open('Model_info/experiment.json','r') as f:
    data=json.load(f)
run_id=data['run_id']

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

vectorizer = load_vectorizer(run_id)

# ------------------ Preprocessing ------------------
def preprocessing(comments):
    cleaned_comments = []

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for comment in comments:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^a-zA-Z ]', '', comment)
        
        tokens = comment.split()
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        cleaned_comments.append(" ".join(tokens))

    return vectorizer.transform(cleaned_comments)




# ------------------ Prediction API ------------------
@app.route('/')
def Hello():
    return jsonify({'name':'Hello'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    comments = data.get("comments", [])

    if not comments:
        return jsonify({'error': 'Zero value received'}), 400

    # Preprocess list of strings
    clean_comments = preprocessing(comments)

    # Predict using MLflow model
    preds = model.predict(clean_comments).tolist()

    # Combine comment + prediction
    results = [
        {"comment": original, "sentiment": pred}
        for original, pred in zip(comments, preds)
    ]

    return jsonify(results)



# ------------------ Run Flask ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)




