# 🚀 Fraud Detection MLOps System

An end-to-end production-style fraud detection system built with:

- FastAPI
- XGBoost
- SHAP Explainability
- MLflow Experiment Tracking
- Drift Detection
- Docker (Multi-container setup)

This project focuses not just on model accuracy, but on **operationalizing machine learning systems**.

---

## 🎯 Problem Context

The dataset contains only **0.17% fraud cases**, making it highly imbalanced.

Instead of optimizing only ROC-AUC (~0.97), this system focuses on:

- Threshold tuning
- Precision vs Recall tradeoffs
- False positive reduction
- Explainability
- Monitoring

---

## 📊 Model Performance

| Metric        | Value |
|---------------|--------|
| ROC-AUC       | ~0.968 |
| Best Threshold| 0.9238 |
| Precision     | ~94%   |
| Recall        | ~82%   |
| False Positives | 5    |

Threshold was optimized using F1-score from the Precision-Recall curve.

---

## 🏗️ System Architecture

```mermaid
flowchart TD

    A[Client - Swagger / HTTP Request] --> B[FastAPI Inference Service]

    B --> C1[XGBoost Model (fraud_xgboost.pkl)]
    B --> C2[SHAP Explainer (TreeExplainer)]
    B --> C3[Drift Monitor (Z-Score Check)]

    C1 --> D[MLflow Tracking (Experiment Logging)]

    subgraph Docker Environment
        B
        C1
        C2
        C3
        D
    end
```
## ⚙️ API Endpoints

| Endpoint | Description |
|-----------|-------------|
| `/predict` | Predict fraud probability |
| `/predict_batch` | Batch prediction |
| `/explain` | SHAP explainability |
| `/metrics` | Model performance metrics |
| `/drift_check` | Feature distribution drift detection |

Swagger Docs:
```
http://localhost:8000/docs
```

---

## 🧠 Key Engineering Decisions

### 1️⃣ Imbalanced Learning
Used `scale_pos_weight` in XGBoost to handle extreme class imbalance.

### 2️⃣ Threshold Optimization
Instead of default 0.5, optimized threshold to 0.9238 for better precision-recall tradeoff.

### 3️⃣ Explainability
Integrated SHAP TreeExplainer for feature-level contribution analysis.

### 4️⃣ Drift Detection
Implemented feature drift detection using Z-score comparison against training distribution.

### 5️⃣ Experiment Tracking
Tracked model training and metrics using MLflow.

### 6️⃣ Containerization
Dockerized API and UI using Docker Compose for reproducible deployment.

---

## 🐳 Running with Docker

```bash
docker compose up --build
```

API will be available at:

```
http://localhost:8000
```

---

## 📁 Project Structure

```
fraud-mlops-system/
│
├── app/
│   ├── api/           # FastAPI service
│   └── ui/            # Streamlit dashboard (WIP)
│
├── training/          # Model training scripts
├── monitoring/        # Drift detection logic
├── models/            # Saved model artifacts
├── docker-compose.yml
└── Dockerfile
```

---

## 🚀 Future Improvements

- Prometheus monitoring
- Logging middleware
- CI/CD pipeline (GitHub Actions)
- Cloud deployment (AWS / Render)
- Model versioning strategy
- Real-time streaming integration

---

## 📌 Why This Project Matters

Most ML projects stop at model training.

This project focuses on:

- Deployment
- Monitoring
- Explainability
- Threshold optimization
- System design

It demonstrates production-minded machine learning engineering.

---

## 👤 Author

Raj Karthik  
ML / Data / AI Enthusiast 