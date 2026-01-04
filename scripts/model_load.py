import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('https://dagshub.com/Rocky0412/Sentiment_Analysis_of_Social_Media_Comments.mlflow')

def load_model_from_model_registry(model_name,version):
    model_uri=f"models:/{model_name}/{version}"
    model=mlflow.pyfunc.load_model(model_uri=model_uri)
    return model

model=load_model_from_model_registry('sentiment_analysis_model',"Staging")
print('Model loaded Successfully')