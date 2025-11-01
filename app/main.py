# app/main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
import numpy as np

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8100")
MLFLOW_MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME", "iris_classifier")
MLFLOW_MODEL_STAGE  = os.getenv("MLFLOW_MODEL_STAGE", "Production")  # or "Staging"

app = FastAPI(title="Iris Prediction API", version="1.0.0")

model = None
class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0)
    sepal_width:  float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width:  float = Field(..., ge=0)

@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # TIP: pin to a stage; change to 'models:/name/version' if you want a specific version
    uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(uri)

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/readyz")
def ready():
    return {"model_loaded": model is not None}

@app.post("/predict")
def predict(x: IrisInput):
    arr = np.array([[x.sepal_length, x.sepal_width, x.petal_length, x.petal_width]])
    pred = model.predict(arr).tolist()[0]
    # try proba if available
    proba = None
    try:
        proba = model.predict_proba(arr).tolist()[0]
    except Exception:
        pass
    return {"prediction": pred, "proba": proba}
