"""FastAPI service for serving the churn-prediction model."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Make `train.py` importable so we can reuse clean_features (same preprocessing as training)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import clean_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_pipeline.joblib"

app = FastAPI(title="Churn Prediction API", version="1.0")

# Load the trained pipeline once at startup (not per request)
model = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    """One customer record - matches the raw dataset columns the model expects."""
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    """Liveness check for load balancers / orchestrators."""
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    """Predict churn for a single customer."""
    # Pydantic object -> one-row DataFrame with the expected column names
    df = pd.DataFrame([customer.model_dump()])
    df = clean_features(df)  # same cleaning as training (coerce TotalCharges, etc.)

    proba = float(model.predict_proba(df)[0, 1])
    prediction = proba >= 0.5

    return {
        "churn": bool(prediction),
        "churn_probability": round(proba, 4),
    }
