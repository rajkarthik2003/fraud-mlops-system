# ==========================================================
# Fraud Detection Baseline - Logistic Regression
# With Scaling, MLflow Tracking, and Threshold Optimization
# ==========================================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ----------------------------------------------------------
# 2. Train-Test Split (Stratified)
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------
# 3. Build Pipeline (Scaling + Logistic Regression)
# ----------------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

# ----------------------------------------------------------
# 4. MLflow Setup
# ----------------------------------------------------------

mlflow.set_experiment("fraud_baseline_logreg")

with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 2000)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("solver", "lbfgs")

    # ------------------------------------------------------
    # 5. Train Model
    # ------------------------------------------------------

    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)

    print("\nClassification Report (Default 0.5 Threshold)")
    print(classification_report(y_test, y_pred))

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)

    # ------------------------------------------------------
    # 7. Threshold Optimization
    # ------------------------------------------------------

    thresholds = np.arange(0.1, 0.95, 0.05)

    best_f1 = 0
    best_threshold = 0.5

    print("\n--- Threshold Tuning ---")

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)

        precision = precision_score(y_test, y_pred_t)
        recall = recall_score(y_test, y_pred_t)
        f1 = f1_score(y_test, y_pred_t)

        print(
            f"Threshold: {t:.2f} | "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f}"
        )

        mlflow.log_metric(f"precision_at_{t:.2f}", precision)
        mlflow.log_metric(f"recall_at_{t:.2f}", recall)
        mlflow.log_metric(f"f1_at_{t:.2f}", f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print("\nBest Threshold:", best_threshold)
    print("Best F1:", best_f1)

    mlflow.log_param("best_threshold", best_threshold)
    mlflow.log_metric("best_f1", best_f1)

    # ------------------------------------------------------
    # 8. Log Model Artifact
    # ------------------------------------------------------

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="fraud_logistic_regression_pipeline"
    )

print("\nTraining complete.")
