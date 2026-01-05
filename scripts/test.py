<<<<<<< HEAD
from mlflow.tracking import MlflowClient
import mlflow

mlflow_uri = "http://ec2-13-232-51-26.ap-south-1.compute.amazonaws.com:8000"
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)

client = MlflowClient()
model = client.get_registered_model("sentiment_analysis_model")

for v in model.latest_versions:
    print("\n=== Version:", v.version, "===")
    print("Run ID:", v.run_id)
    print("Source:", v.source)
=======
import mlflow

# Correct S3 path (check the 'model/' folder exists in S3)
model_uri = "s3://mlflow-s3-artifact/105166158212028233/de5501699b0f463eae2ed76d75322b03/artifacts/model"

model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded successfully")
>>>>>>> eb0dce0
