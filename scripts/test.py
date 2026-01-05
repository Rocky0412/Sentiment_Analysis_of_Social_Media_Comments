import mlflow

# Correct S3 path (check the 'model/' folder exists in S3)
model_uri = "s3://mlflow-s3-artifact/105166158212028233/de5501699b0f463eae2ed76d75322b03/artifacts/model"

model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded successfully")
