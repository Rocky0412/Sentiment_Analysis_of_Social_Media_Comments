import mlflow
import random

mlflow_uri = "http://ec2-13-232-51-26.ap-south-1.compute.amazonaws.com:8000"

# THIS is required for logging runs & metrics
mlflow.set_tracking_uri(mlflow_uri)

# (Optional) This is only for model registry
mlflow.set_registry_uri(mlflow_uri)

mlflow.set_experiment("Test")

with mlflow.start_run() as run:
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", random.random())
    print("Run ID:", run.info.run_id)

client = mlflow.tracking.MlflowClient()
print(client)

   
