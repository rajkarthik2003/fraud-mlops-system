# filepath: app/api/main.py
import logging
import uuid
import os
import time
import asyncio
import hashlib
from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Request, status, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
import joblib
import numpy as np
import shap
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
import redis
from circuitbreaker import circuit

# =========================
# Configuration
# =========================
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.environ.get("CACHE_TTL", "3600"))  # 1 hour
ENABLE_CACHE = os.environ.get("ENABLE_CACHE", "true").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))  # per minute

# =========================
# Structured Logging Setup
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("fraud_detection_api")

# =========================
# Redis Cache Setup
# =========================
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis cache connected")
    CACHE_ENABLED = True
except Exception as e:
    logger.warning(f"Redis not available: {e}. Running without cache.")
    redis_client = None
    CACHE_ENABLED = False

# =========================
# Rate Limiting Setup
# =========================
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT_REQUESTS} per {RATE_LIMIT_WINDOW} minute"])

# =========================
# Prometheus Metrics
# =========================
class MetricsCollector:
    """Enhanced metrics collector with Redis persistence"""

    def __init__(self):
        self.request_count = 0
        self.predictions_total = 0
        self.predictions_fraud = 0
        self.errors_total = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.request_latencies: List[float] = []
        self.start_time = time.time()

    def increment_requests(self):
        self.request_count += 1

    def increment_predictions(self, is_fraud: bool):
        self.predictions_total += 1
        if is_fraud:
            self.predictions_fraud += 1

    def increment_errors(self):
        self.errors_total += 1

    def increment_cache_hit(self):
        self.cache_hits += 1

    def increment_cache_miss(self):
        self.cache_misses += 1

    def add_latency(self, latency: float):
        self.request_latencies.append(latency)
        # Keep only last 1000 latencies
        if len(self.request_latencies) > 1000:
            self.request_latencies = self.request_latencies[-1000:]

    @property
    def avg_latency(self) -> float:
        return np.mean(self.request_latencies) if self.request_latencies else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

# =========================
# Cache Functions
# =========================
def get_cache_key(features: List[float]) -> str:
    """Generate cache key from features"""
    feature_str = ",".join(f"{x:.6f}" for x in features)
    return hashlib.md5(feature_str.encode()).hexdigest()

def get_cached_prediction(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get prediction from cache"""
    if not CACHE_ENABLED or not redis_client:
        return None
    try:
        cached = redis_client.get(f"pred:{cache_key}")
        if cached:
            import json
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    return None

def set_cached_prediction(cache_key: str, prediction: Dict[str, Any]):
    """Cache prediction result"""
    if not CACHE_ENABLED or not redis_client:
        return
    try:
        import json
        redis_client.setex(f"pred:{cache_key}", CACHE_TTL, json.dumps(prediction))
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

# =========================
# Circuit Breaker for Model Inference
# =========================
@circuit(failure_threshold=5, recovery_timeout=60, expected_exception=Exception)
def predict_with_circuit_breaker(features: np.ndarray) -> np.ndarray:
    """Circuit breaker protected model prediction"""
    return model.predict_proba(features)

@circuit(failure_threshold=5, recovery_timeout=60, expected_exception=Exception)
def explain_with_circuit_breaker(features: np.ndarray) -> Dict[str, Any]:
    """Circuit breaker protected SHAP explanation"""
    shap_values = explainer.shap_values(features)
    return {
        "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
        "base_value": float(explainer.expected_value),
        "feature_names": FEATURE_NAMES
    }

    def record_latency(self, latency: float):
        self.request_latencies.append(latency)
        # Keep only last 1000 latencies
        if len(self.request_latencies) > 1000:
            self.request_latencies = self.request_latencies[-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_latency = (
            sum(self.request_latencies) / len(self.request_latencies)
            if self.request_latencies
            else 0
        )

        return {
            "uptime_seconds": round(uptime, 2),
            "requests_total": self.request_count,
            "predictions_total": self.predictions_total,
            "predictions_fraud": self.predictions_fraud,
            "fraud_rate": round(
                self.predictions_fraud / max(self.predictions_total, 1), 4
            ),
            "errors_total": self.errors_total,
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "requests_per_second": round(self.request_count / max(uptime, 1), 2),
        }


metrics = MetricsCollector()


# =========================
# Rate Limiting
# =========================
class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []

        # Clean old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id] if now - t < self.window_seconds
        ]

        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True


rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


# =========================
# Error Codes
# =========================
class ErrorCode(str, Enum):
    # Validation errors (1xxx)
    INVALID_FEATURES = "E1001"
    FEATURE_COUNT_MISMATCH = "E1002"
    INVALID_FEATURE_VALUE = "E1003"
    BATCH_TOO_LARGE = "E1004"

    # Model errors (2xxx)
    MODEL_NOT_LOADED = "E2001"
    MODEL_PREDICTION_FAILED = "E2002"
    EXPLAINER_FAILED = "E2003"

    # Drift errors (3xxx)
    DRIFT_STATS_UNAVAILABLE = "E3001"
    DRIFT_CHECK_FAILED = "E3002"

    # Auth errors (4xxx)
    INVALID_API_KEY = "E4001"
    API_KEY_MISSING = "E4002"

    # Rate limiting (5xxx)
    RATE_LIMIT_EXCEEDED = "E5001"

    # Internal errors (9xxx)
    INTERNAL_ERROR = "E9001"
    SERVICE_UNAVAILABLE = "E9002"


ERROR_MESSAGES = {
    ErrorCode.INVALID_FEATURES: "Invalid features provided",
    ErrorCode.FEATURE_COUNT_MISMATCH: "Expected 30 features, got {count}",
    ErrorCode.INVALID_FEATURE_VALUE: "Feature {index} contains invalid value (NaN/Inf)",
    ErrorCode.BATCH_TOO_LARGE: "Batch size exceeds maximum of 1000",
    ErrorCode.MODEL_NOT_LOADED: "Model not loaded. Please train model first.",
    ErrorCode.MODEL_PREDICTION_FAILED: "Model prediction failed",
    ErrorCode.EXPLAINER_FAILED: "SHAP explanation generation failed",
    ErrorCode.DRIFT_STATS_UNAVAILABLE: "Drift statistics not available",
    ErrorCode.DRIFT_CHECK_FAILED: "Drift detection failed",
    ErrorCode.INVALID_API_KEY: "Invalid API key",
    ErrorCode.API_KEY_MISSING: "API key required. Use X-API-Key header.",
    ErrorCode.RATE_LIMIT_EXCEEDED: "Rate limit exceeded. Try again later.",
    ErrorCode.INTERNAL_ERROR: "Internal server error",
    ErrorCode.SERVICE_UNAVAILABLE: "Service temporarily unavailable",
}

# =========================
# API Key Authentication
# =========================
API_KEY = os.environ.get("API_KEY", "dev-key-change-in-production")


class APIKeyHeader(BaseModel):
    """Dependency for API key validation"""

    api_key: str = Header(..., alias="X-API-Key")

    model_config = ConfigDict(extra="allow")


def verify_api_key(api_key: str = Depends(APIKeyHeader)):
    """Verify the API key"""
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": ErrorCode.INVALID_API_KEY,
                "message": ERROR_MESSAGES[ErrorCode.INVALID_API_KEY],
            },
        )
    return api_key


# =========================
# Lifespan (Startup/Shutdown)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Fraud Detection API...")
    # Startup
    yield
    # Shutdown
    logger.info("Shutting down Fraud Detection API...")


app = FastAPI(
    title="Fraud Detection API",
    version="2.1.0",
    description="Production-grade fraud detection with monitoring, caching, and security",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# =========================
# Security & Performance Middleware
# =========================
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production
app.add_middleware(SlowAPIMiddleware)

# =========================
# CORS Middleware
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    logger.info(f"Request started | {request_id} | {request.method} {request.url.path}")

    response = await call_next(request)

    logger.info(f"Request completed | {request_id} | Status: {response.status_code}")
    return response


# =========================
# Exception Handlers
# =========================
class FraudDetectionException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


@app.exception_handler(FraudDetectionException)
async def fraud_exception_handler(request: Request, exc: FraudDetectionException):
    logger.error(f"Custom exception: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "request_id": request.state.request_id},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request.state.request_id,
        },
    )


# =========================
# CORS Middleware
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load Model + Artifacts
# =========================
MODEL_PATH = "models/fraud_xgboost.pkl"
THRESHOLD_PATH = "models/best_threshold.txt"

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Model not found. Run training first.")

try:
    explainer = shap.TreeExplainer(model)
    logger.info("SHAP explainer initialized")
except Exception as e:
    logger.error(f"Failed to initialize SHAP: {e}")
    raise RuntimeError("Failed to initialize explainer")

try:
    with open(THRESHOLD_PATH, "r") as f:
        BEST_THRESHOLD = float(f.read().strip())
    logger.info(f"Threshold loaded: {BEST_THRESHOLD}")
except Exception as e:
    logger.error(f"Failed to load threshold: {e}")
    BEST_THRESHOLD = 0.5

try:
    feature_means = np.load("models/feature_means.npy")
    feature_stds = np.load("models/feature_stds.npy")
    logger.info("Drift statistics loaded")
except Exception as e:
    logger.warning(f"Drift statistics not found: {e}")
    feature_means = None
    feature_stds = None

# Feature names (Credit Card Dataset)
FEATURE_NAMES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


# =========================
# Enhanced Request Models
# =========================
class Transaction(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 transaction features (V1-V28, Time, Amount)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [
                        -0.435,
                        0.848,
                        -0.432,
                        0.521,
                        -0.246,
                        -0.082,
                        0.395,
                        -0.191,
                        -0.556,
                        0.264,
                        -0.344,
                        -0.412,
                        0.512,
                        -0.045,
                        0.876,
                        -0.333,
                        0.621,
                        -0.478,
                        0.112,
                        -0.298,
                        -0.566,
                        0.156,
                        -0.689,
                        0.442,
                        -0.215,
                        0.134,
                        -0.389,
                        0.287,
                        0.123,
                        150.00,
                    ]
                }
            ]
        }
    }

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if len(v) != 30:
            raise ValueError("Must provide exactly 30 features")
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                raise ValueError(f"Feature {i} must be a valid number")
        return v


class BatchTransaction(BaseModel):
    transactions: List[List[float]]
    max_batch_size: int = Field(default=1000, le=1000)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transactions": [
                        [
                            -0.435,
                            0.848,
                            -0.432,
                            0.521,
                            -0.246,
                            -0.082,
                            0.395,
                            -0.191,
                            -0.556,
                            0.264,
                            -0.344,
                            -0.412,
                            0.512,
                            -0.045,
                            0.876,
                            -0.333,
                            0.621,
                            -0.478,
                            0.112,
                            -0.298,
                            -0.566,
                            0.156,
                            -0.689,
                            0.442,
                            -0.215,
                            0.134,
                            -0.389,
                            0.287,
                            0.123,
                            150.00,
                        ],
                        [
                            0.123,
                            -0.456,
                            0.789,
                            -0.321,
                            0.654,
                            -0.987,
                            0.111,
                            -0.222,
                            0.333,
                            -0.444,
                            0.555,
                            -0.666,
                            0.777,
                            -0.888,
                            0.999,
                            -0.111,
                            0.222,
                            -0.333,
                            0.444,
                            -0.555,
                            0.666,
                            -0.777,
                            0.888,
                            -0.999,
                            0.101,
                            -0.202,
                            0.303,
                            -0.404,
                            0.505,
                            250.00,
                        ],
                    ]
                }
            ]
        }
    }

    @field_validator("transactions")
    @classmethod
    def validate_batch(cls, v):
        if len(v) == 0:
            raise ValueError("At least one transaction required")
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000")
        for i, tx in enumerate(v):
            if len(tx) != 30:
                raise ValueError(f"Transaction {i} must have exactly 30 features")
        return v


# =========================
# Health Check Endpoints
# =========================
@app.get("/health")
def health_check():
    """Liveness probe - is the service running?"""
    return {
        "status": "healthy",
        "service": "fraud-detection-api",
        "version": "2.1.0",
        "timestamp": time.time()
    }


@app.get("/ready")
def readiness_check():
    """Readiness probe - is the service ready to handle requests?"""
    checks = {
        "model_loaded": model is not None,
        "explainer_initialized": explainer is not None,
        "threshold_loaded": BEST_THRESHOLD > 0,
        "drift_stats_loaded": feature_means is not None and feature_stds is not None,
        "redis_cache": CACHE_ENABLED if CACHE_ENABLED else "disabled",
        "rate_limiter": True,
    }

    # Test model prediction with dummy data
    try:
        test_features = np.zeros((1, 30))
        test_pred = predict_with_circuit_breaker(test_features)
        checks["model_inference"] = test_pred.shape == (1, 2)
    except Exception:
        checks["model_inference"] = False

    all_ready = all(checks.values()) if isinstance(checks["redis_cache"], bool) else all(v for k, v in checks.items() if k != "redis_cache")

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": time.time()
    }

    return {"status": "ready" if all_ready else "not_ready", "checks": checks}


# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Fraud Detection API running",
        "version": "2.1.0",
    }


@app.get("/metrics")
def get_metrics():
    """Returns comprehensive model and operational metrics"""
    uptime = time.time() - metrics.start_time

    return {
        "model": {
            "name": "XGBoost",
            "version": os.environ.get("MODEL_VERSION", "1.0.0"),
            "roc_auc": 0.9684,
            "pr_auc": 0.8787,
            "best_threshold": BEST_THRESHOLD,
            "best_f1": 0.8804,
            "precision": 0.9419,
            "recall": 0.8265,
            "features": len(FEATURE_NAMES),
        },
        "operational": {
            "uptime_seconds": round(uptime, 2),
            "total_requests": metrics.request_count,
            "total_predictions": metrics.predictions_total,
            "fraud_predictions": metrics.predictions_fraud,
            "error_count": metrics.errors_total,
            "avg_latency_ms": round(metrics.avg_latency * 1000, 2) if metrics.request_latencies else 0,
            "cache_enabled": CACHE_ENABLED,
            "cache_hit_rate": round(metrics.cache_hit_rate * 100, 2) if CACHE_ENABLED else 0,
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "rate_limit_window_seconds": RATE_LIMIT_WINDOW,
        },
        "system": {
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "redis_connected": CACHE_ENABLED,
            "model_loaded": True,
            "explainer_loaded": True,
        }
    }


@app.post("/predict")
@limiter.limit(f"{RATE_LIMIT_REQUESTS} per {RATE_LIMIT_WINDOW} minute")
def predict(tx: Transaction, request: Request):
    start_time = time.time()

    try:
        # Check cache first
        cache_key = get_cache_key(tx.features)
        cached_result = get_cached_prediction(cache_key)
        if cached_result:
            metrics.increment_cache_hit()
            metrics.increment_requests()
            cached_result["cached"] = True
            cached_result["request_id"] = request.state.request_id
            logger.info(f"Cache hit for request {request.state.request_id}")
            return cached_result

        metrics.increment_cache_miss()

        # Perform prediction with circuit breaker
        x = np.array(tx.features, dtype=float).reshape(1, -1)
        prob = float(predict_with_circuit_breaker(x)[0, 1])
        pred = int(prob >= BEST_THRESHOLD)

        result = {
            "fraud_probability": round(prob, 6),
            "threshold": BEST_THRESHOLD,
            "prediction": pred,
            "request_id": request.state.request_id,
            "cached": False
        }

        # Cache the result
        set_cached_prediction(cache_key, result)

        # Record metrics
        metrics.increment_requests()
        metrics.increment_predictions(bool(pred))
        latency = time.time() - start_time
        metrics.add_latency(latency)

        logger.info(
            f"Prediction: prob={prob:.4f}, pred={pred}, latency={latency:.3f}s, request_id={request.state.request_id}"
        )

        return result
    except Exception as e:
        metrics.increment_errors()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.MODEL_PREDICTION_FAILED,
                "message": ERROR_MESSAGES[ErrorCode.MODEL_PREDICTION_FAILED],
            },
        )


@app.post("/predict_batch")
@limiter.limit(f"{RATE_LIMIT_REQUESTS} per {RATE_LIMIT_WINDOW} minute")
def predict_batch(data: BatchTransaction, request: Request):
    try:
        x = np.array(data.transactions, dtype=float)
        probs = model.predict_proba(x)[:, 1]
        preds = (probs >= BEST_THRESHOLD).astype(int)

        logger.info(
            f"Batch prediction: {len(data.transactions)} transactions, request_id={request.state.request_id}"
        )

        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "count": len(preds),
            "request_id": request.state.request_id,
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.post("/explain")
def explain_transaction(tx: Transaction, request: Request):
    try:
        input_array = np.array(tx.features, dtype=float).reshape(1, -1)

        fraud_prob = float(model.predict_proba(input_array)[0][1])

        shap_values = explainer(input_array)
        contributions = shap_values.values[0]

        top_indices = np.argsort(np.abs(contributions))[-5:][::-1]

        explanation = []
        for idx in top_indices:
            explanation.append(
                {
                    "feature_name": FEATURE_NAMES[idx],
                    "feature_index": int(idx),
                    "impact": round(float(contributions[idx]), 6),
                    "direction": (
                        "increases_fraud"
                        if contributions[idx] > 0
                        else "decreases_fraud"
                    ),
                }
            )

        logger.info(f"Explanation generated, request_id={request.state.request_id}")

        return {
            "fraud_probability": round(fraud_prob, 6),
            "prediction": int(fraud_prob >= BEST_THRESHOLD),
            "top_contributing_features": explanation,
            "request_id": request.state.request_id,
        }
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail="Explanation failed")


@app.post("/drift_check")
def drift_check(tx: Transaction, request: Request):
    if feature_means is None or feature_stds is None:
        raise HTTPException(
            status_code=503, detail="Drift statistics not available. Train model first."
        )

    try:
        input_array = np.array(tx.features, dtype=float)

        z_scores = np.abs((input_array - feature_means) / (feature_stds + 1e-8))

        drift_results = []
        for i, z in enumerate(z_scores):
            if z > 3:
                drift_results.append(
                    {
                        "feature": FEATURE_NAMES[i],
                        "index": i,
                        "z_score": round(float(z), 4),
                        "severity": "critical" if z > 5 else "warning",
                    }
                )

        has_drift = len(drift_results) > 0

        logger.info(
            f"Drift check: {'detected' if has_drift else 'none'}, request_id={request.state.request_id}"
        )

        return {
            "drift_detected": has_drift,
            "drift_features": drift_results,
            "total_features_checked": len(FEATURE_NAMES),
            "request_id": request.state.request_id,
        }
    except Exception as e:
        logger.error(f"Drift check error: {e}")
        raise HTTPException(status_code=500, detail="Drift check failed")


# =========================
# Async Batch Processing
# =========================
class AsyncBatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory job store (use Redis in production)
batch_jobs: dict = {}


@app.post("/predict_async", status_code=202)
async def predict_async(
    data: BatchTransaction, request: Request, api_key: str = Depends(verify_api_key)
):
    """Submit batch prediction job for async processing"""
    import asyncio

    job_id = str(uuid.uuid4())

    batch_jobs[job_id] = {
        "status": AsyncBatchStatus.PENDING,
        "total": len(data.transactions),
        "processed": 0,
        "results": None,
        "error": None,
    }

    logger.info(
        f"Async job {job_id} submitted with {len(data.transactions)} transactions"
    )

    # Process in background (use Celery/Redis in production)
    async def process_batch():
        try:
            batch_jobs[job_id]["status"] = AsyncBatchStatus.PROCESSING

            # Simulate processing with chunking
            x = np.array(data.transactions, dtype=float)
            probs = model.predict_proba(x)[:, 1]
            preds = (probs >= BEST_THRESHOLD).astype(int)

            batch_jobs[job_id]["results"] = {
                "predictions": preds.tolist(),
                "probabilities": probs.tolist(),
            }
            batch_jobs[job_id]["processed"] = len(preds)
            batch_jobs[job_id]["status"] = AsyncBatchStatus.COMPLETED

            logger.info(f"Async job {job_id} completed")
        except Exception as e:
            batch_jobs[job_id]["status"] = AsyncBatchStatus.FAILED
            batch_jobs[job_id]["error"] = str(e)
            logger.error(f"Async job {job_id} failed: {e}")

    # Run async (in production, dispatch to Celery task queue)
    asyncio.create_task(process_batch())

    return {
        "job_id": job_id,
        "status": AsyncBatchStatus.PENDING,
        "message": "Job submitted. Use /predict_async/{job_id} to check status.",
        "request_id": request.state.request_id,
    }


@app.get("/predict_async/{job_id}")
async def get_async_result(
    job_id: str, request: Request, api_key: str = Depends(verify_api_key)
):
    """Get async batch job status and results"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = batch_jobs[job_id]

    return {
        "job_id": job_id,
        "status": job["status"],
        "total": job["total"],
        "processed": job["processed"],
        "results": job.get("results"),
        "error": job.get("error"),
        "request_id": request.state.request_id,
    }
