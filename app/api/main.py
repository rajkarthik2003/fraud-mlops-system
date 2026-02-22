from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import shap

app = FastAPI(title="Fraud Detection API", version="1.0")

# =========================
# Load Model + Artifacts
# =========================

MODEL_PATH = "models/fraud_xgboost.pkl"
THRESHOLD_PATH = "models/best_threshold.txt"

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

with open(THRESHOLD_PATH, "r") as f:
    BEST_THRESHOLD = float(f.read().strip())

feature_means = np.load("models/feature_means.npy")
feature_stds = np.load("models/feature_stds.npy")

# Feature names (Credit Card Dataset)
FEATURE_NAMES = [
    "Time",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "Amount"
]

# =========================
# Request Models
# =========================

class Transaction(BaseModel):
    features: List[float]

# =========================
# Routes
# =========================

@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud Detection API running"}

@app.get("/metrics")
def get_metrics():
    return {
        "model": "XGBoost",
        "roc_auc": 0.9684,
        "pr_auc": 0.8787,
        "best_threshold": 0.9238,
        "best_f1": 0.8804,
        "precision": 0.9419,
        "recall": 0.8265
    }

@app.post("/predict")
def predict(tx: Transaction):
    x = np.array(tx.features, dtype=float).reshape(1, -1)
    prob = float(model.predict_proba(x)[0, 1])
    pred = int(prob >= BEST_THRESHOLD)

    return {
        "fraud_probability": prob,
        "threshold": BEST_THRESHOLD,
        "prediction": pred
    }

@app.post("/predict_batch")
def predict_batch(data: List[List[float]]):
    x = np.array(data, dtype=float)
    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= BEST_THRESHOLD).astype(int)

    return {
        "predictions": preds.tolist()
    }

@app.post("/explain")
def explain_transaction(tx: Transaction):

    input_array = np.array(tx.features, dtype=float).reshape(1, -1)

    fraud_prob = float(model.predict_proba(input_array)[0][1])

    shap_values = explainer(input_array)
    contributions = shap_values.values[0]

    top_indices = np.argsort(np.abs(contributions))[-5:][::-1]

    explanation = []
    for idx in top_indices:
        explanation.append({
            "feature_name": FEATURE_NAMES[idx],
            "impact": float(contributions[idx])
        })

    return {
        "fraud_probability": fraud_prob,
        "top_contributing_features": explanation
    }

@app.post("/drift_check")
def drift_check(tx: Transaction):

    input_array = np.array(tx.features, dtype=float)

    z_scores = np.abs((input_array - feature_means) / feature_stds)

    drift_features = [
        FEATURE_NAMES[i]
        for i, z in enumerate(z_scores)
        if z > 3
    ]

    return {
        "drift_detected": len(drift_features) > 0,
        "drift_features": drift_features
    }