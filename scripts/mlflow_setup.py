import mlflow
import random


mlflow.set_registry_uri(uri='http://127.0.0.1:5000')
mlflow.set_experiment('Test')

with mlflow.start_run() as run:
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", random.random())
    print("Run ID:", run.info.run_id)
    client=mlflow.client.MlflowClient()
    print(f'{client}')
   
