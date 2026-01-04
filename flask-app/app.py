from flask import Flask
import mlflow
from mlflow.pyfunc import load_model

# Set MLflow tracking URI
mlflow.set_tracking_uri('https://dagshub.com/Rocky0412/Sentiment_Analysis_of_Social_Media_Comments.mlflow')

app = Flask(__name__)

@app.route('/')
def helloworld():
    model_name = "sentiment_analysis_model"       # <-- Change to your registered model name
    model_version = 'latest'                 # <-- Your model version
    
    model_uri = f"models:/{model_name}/{model_version}"
    
    return f"Model URI: {model_uri}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
