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