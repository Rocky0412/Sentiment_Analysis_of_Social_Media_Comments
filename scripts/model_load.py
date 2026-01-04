import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://ec2-13-232-51-26.ap-south-1.compute.amazonaws.com:8000")

run_id = "ac4746e981c141f3b710fb6c3ea5a470"
artifact_path = "model"  # same as used in log_model

model_uri = f"runs:/{run_id}/{artifact_path}"
model = mlflow.sklearn.load_model(model_uri)



