# 🛡️ Fraud Detection System with MLOps Tracking

A production-style fraud detection pipeline built with structured experimentation, reproducible training, and MLflow experiment tracking.

This project demonstrates how to move from a simple ML baseline to a system designed with engineering rigor and MLOps foundations.

---

## 🚀 Project Overview

This system detects fraudulent credit card transactions using a Logistic Regression baseline wrapped inside a proper preprocessing pipeline and tracked with MLflow.

Unlike notebook-only projects, this implementation focuses on:

- Clean training pipeline
- Feature scaling
- Imbalanced classification evaluation
- Experiment tracking
- Reproducibility
- Production-aware metric interpretation

---

## 📂 Project Structure

fraud-mlops-system/

│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── training/
│   └── train_baseline.py
│
├── mlruns/
├── models/
└── README.md

---

## 🧠 Problem Context

Fraud detection is an extremely imbalanced classification problem:

- Legitimate transactions: ~99.8%
- Fraud transactions: ~0.2%

Accuracy alone is misleading.

This project evaluates using:

- ROC-AUC
- PR-AUC
- Precision / Recall
- Minority class performance

---

## ⚙️ Baseline Model

Model:
- Logistic Regression
- StandardScaler (feature normalization)
- Class balancing enabled
- Implemented using Scikit-learn Pipeline

Why Logistic Regression?

- Strong tabular baseline
- Interpretable coefficients
- Fast training
- Stable optimization
- Reliable calibration

---

## 📊 Final Reproducible Baseline Results (MLflow Tracked)

ROC-AUC: 0.9720  
PR-AUC: 0.7189  
Recall (Fraud): 0.92  
Precision (Fraud): 0.06  
Accuracy: 0.98  

Classification Report:

precision    recall  f1-score   support

0       1.00      0.98      0.99     56864  
1       0.06      0.92      0.11        98  

---

## 🔍 Interpretation

- The model detects 92% of fraud cases.
- Precision is low due to extreme imbalance.
- PR-AUC is prioritized over accuracy.
- The baseline intentionally favors recall over precision.

This mirrors real-world fraud systems where missing fraud is more costly than false alarms.

---

## 📈 Experiment Tracking (MLflow)

All experiments are tracked using MLflow:

- Parameters
- Metrics
- Model artifacts
- Reproducible runs

Start MLflow UI:

python -m mlflow ui

Open:

http://127.0.0.1:5000

Experiment name:

fraud_baseline_logreg

---

## 🔄 Reproducibility

To reproduce the experiment:

python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt  
python training/train_baseline.py  

Then:

python -m mlflow ui  

---

## 🧱 MLOps Foundations Implemented

- Script-based training (not notebook-only)
- Clean project structure
- Experiment tracking
- Metric logging
- Model artifact logging
- Local MLflow tracking backend
- Reproducible pipeline

---

## 📌 Design Tradeoff

This baseline intentionally prioritizes recall over precision, accepting more false positives in order to minimize missed fraud.

Threshold tuning is the next step for business-level optimization.

---

## 🔮 Planned Extensions

- Decision threshold optimization
- Model comparison (XGBoost / LightGBM)
- Model registry usage
- CI/CD integration
- Docker deployment
- Real-time inference API
- Monitoring & drift detection

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- MLflow
- Pandas
- NumPy
- Matplotlib
- SQLite (MLflow backend)

---

## 🎯 Key Takeaway

This project demonstrates how to turn a baseline ML model into a structured, trackable, reproducible ML system with MLOps foundations.
