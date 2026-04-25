# 🚀 Fraud Detection MLOps System v2.1.0

An enterprise-grade, production-ready fraud detection system with advanced MLOps capabilities.

## ✨ Features

### 🤖 ML & AI
- **XGBoost Model**: Optimized for imbalanced fraud detection (0.17% fraud rate)
- **SHAP Explainability**: Feature importance and prediction explanations
- **Drift Detection**: Automated monitoring of data distribution changes
- **Threshold Optimization**: F1-score optimized decision boundary

### ⚡ Performance & Scalability
- **Redis Caching**: Intelligent prediction caching with TTL
- **Circuit Breaker**: Fault-tolerant model inference with automatic recovery
- **Rate Limiting**: Configurable request throttling (SlowAPI)
- **Gzip Compression**: Optimized response compression
- **Multi-worker**: Uvicorn with 4 workers for concurrent requests

### 🔒 Security & Reliability
- **API Key Authentication**: Secure endpoint access
- **Input Validation**: Pydantic-powered request validation
- **Health Checks**: Comprehensive liveness and readiness probes
- **Error Handling**: Structured error responses with error codes
- **Request ID Tracking**: End-to-end request tracing

### 📊 Monitoring & Observability
- **Prometheus Metrics**: Real-time performance monitoring
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Operational Metrics**: Cache hit rates, latency tracking, error counts
- **MLflow Integration**: Experiment tracking and model versioning

### 🐳 DevOps & Deployment
- **Docker**: Multi-service containerized deployment
- **Kubernetes**: Production-ready K8s manifests with HPA
- **PostgreSQL**: Prediction logging and audit trail
- **CI/CD**: GitHub Actions with automated testing and linting

---

## 🎯 Problem Context

The dataset contains only **0.17% fraud cases**, making it highly imbalanced.

Instead of optimizing only ROC-AUC (~0.97), this system focuses on:

- Threshold tuning for optimal F1-score
- Precision vs Recall tradeoffs
- False positive reduction
- Explainability for regulatory compliance
- Monitoring for production reliability

---

## 📊 Model Performance

| Metric        | Value  | Description |
|---------------|--------|-------------|
| ROC-AUC       | 0.968  | Overall discriminative ability |
| Best Threshold| 0.924  | F1-optimized decision boundary |
| Precision     | 94.2%  | True positive rate |
| Recall        | 82.6%  | Fraud detection rate |
| F1-Score      | 88.0%  | Harmonic mean of precision/recall |

**Threshold optimized using F1-score from Precision-Recall curve for imbalanced classification.**

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Client Request] --> B[API Gateway / Load Balancer]
    B --> C[FastAPI Service]
    C --> D{Redis Cache}
    D --> E[Model Inference]
    E --> F[XGBoost Model]
    F --> G[SHAP Explainer]
    C --> H[PostgreSQL]
    C --> I[MLflow]
    C --> J[Prometheus]

    K[Streamlit UI] --> C
    L[Monitoring] --> J
```

### Service Components

- **API Service**: FastAPI with ML inference, caching, and monitoring
- **UI Dashboard**: Streamlit interface for model interaction
- **MLflow**: Experiment tracking and model registry
- **PostgreSQL**: Prediction logging and analytics
- **Redis**: High-performance caching layer
- **Prometheus**: Metrics collection and alerting

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- 8GB+ RAM recommended

### 1. Clone & Setup
```bash
git clone https://github.com/rajkarthik2003/fraud-mlops-system.git
cd fraud-mlops-system
```

### 2. Download Dataset
```bash
# Download creditcard.csv from Kaggle
# Place in data/ directory
mkdir -p data
# Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

### 3. Train Model
```bash
python training/export_best_model.py
```

### 4. Launch Services
```bash
docker-compose up --build
```

### 5. Access Services
- **API**: http://localhost:8000/docs (Swagger UI)
- **UI**: http://localhost:8501 (Streamlit Dashboard)
- **MLflow**: http://localhost:5001 (Experiment Tracking)
- **Prometheus**: http://localhost:9090 (Metrics)
- **API Health**: http://localhost:8000/health

---

## 📡 API Endpoints

### Core Endpoints
- `POST /predict` - Single prediction with caching
- `POST /predict_batch` - Batch predictions
- `POST /explain` - SHAP feature explanations
- `POST /drift_check` - Data drift detection

### Monitoring Endpoints
- `GET /health` - Liveness probe
- `GET /ready` - Readiness probe
- `GET /metrics` - Prometheus metrics
- `GET /` - API information

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-in-production" \
  -d '{
    "features": [0.0, -1.5, 0.5, -0.8, 0.2, 0.1, -0.3, 0.4, -0.2, 0.6,
                 -0.1, 0.3, -0.4, 0.7, -0.5, 0.8, -0.6, 0.9, -0.7, 1.0,
                 -0.8, 0.2, -0.9, 0.3, -1.0, 0.4, -0.1, 0.5, -0.2, 250.0]
  }'
```

---

## 🔧 Configuration

### Environment Variables
```bash
# Security
API_KEY=your-secure-api-key
CORS_ORIGINS=http://localhost:8501,https://yourdomain.com

# Performance
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600
ENABLE_CACHE=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60

# Monitoring
MLFLOW_TRACKING_URI=http://localhost:5001
DATABASE_URL=postgresql://user:pass@localhost:5432/fraud_db
```

### Scaling Configuration
- **API Workers**: 4 uvicorn workers
- **Kubernetes**: HPA with CPU/memory scaling
- **Redis**: Persistent caching with TTL
- **Rate Limits**: Configurable per endpoint

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=app

# Run with coverage report
pytest tests/ --cov=app --cov-report=html

# Run specific test
pytest tests/test_api.py::TestTransactionValidation::test_valid_transaction -v
```

**Test Coverage**: 50%+ with mocking for external dependencies

---

## 📦 Deployment

### Docker Compose (Development)
```bash
docker-compose up --build -d
docker-compose logs -f api
```

### Kubernetes (Production)
```bash
kubectl apply -f k8s/
kubectl get pods -n fraud-detection
```

### Production Checklist
- [ ] Change API key from default
- [ ] Configure CORS origins
- [ ] Set up Redis persistence
- [ ] Configure monitoring alerts
- [ ] Set up log aggregation
- [ ] Configure backup strategies

---

## 📈 Monitoring & Metrics

### Key Metrics
- **Request Latency**: P95 < 100ms
- **Cache Hit Rate**: > 80% for repeated requests
- **Error Rate**: < 1%
- **Model Accuracy**: ROC-AUC > 0.96

### Health Checks
- **Liveness**: `/health` - Service availability
- **Readiness**: `/ready` - Dependency health
- **Metrics**: `/metrics` - Prometheus format

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `pytest tests/`
4. Format code: `black app/ && flake8 app/`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle
- **Libraries**: XGBoost, SHAP, FastAPI, Redis, Prometheus
- **Inspiration**: Real-world MLOps best practices

---

**⭐ Star this repo if you find it useful!**

    B --> C1[XGBoost Model - fraud_xgboost_pkl]
    B --> C2[SHAP Explainer - TreeExplainer]
    B --> C3[Drift Monitor - Z Score Check]

    C1 --> D[MLflow Tracking - Experiment Logging]

    subgraph Docker_Environment
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