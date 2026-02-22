import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("data/creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------------
# MLflow Experiment
# -------------------------------
mlflow.set_experiment("fraud_xgboost")

with mlflow.start_run():

    # -------------------------------
    # Train model
    # -------------------------------
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", float(scale_pos_weight))

    model.fit(X_train, y_train)

    # -------------------------------
    # Probabilities + global metrics
    # -------------------------------
    y_probs = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)

    # -------------------------------
    # Threshold Optimization (best F1)
    # -------------------------------
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    best_precision = float(precision[best_idx])
    best_recall = float(recall[best_idx])

    print("\n----- Threshold Optimization -----")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Precision at Best Threshold: {best_precision:.4f}")
    print(f"Recall at Best Threshold: {best_recall:.4f}")

    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("best_f1", best_f1)
    mlflow.log_metric("best_precision", best_precision)
    mlflow.log_metric("best_recall", best_recall)

    # -------------------------------
    # Confusion Matrix @ best threshold
    # -------------------------------
    y_pred_best = (y_probs >= best_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y_test, y_pred_best)
    rec = recall_score(y_test, y_pred_best)
    f1 = f1_score(y_test, y_pred_best)

    print("\n----- Confusion Matrix @ Best Threshold -----")
    print(cm)
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")

    mlflow.log_metric("tn", int(tn))
    mlflow.log_metric("fp", int(fp))
    mlflow.log_metric("fn", int(fn))
    mlflow.log_metric("tp", int(tp))

    mlflow.log_metric("precision_at_best_threshold", float(prec))
    mlflow.log_metric("recall_at_best_threshold", float(rec))
    mlflow.log_metric("f1_at_best_threshold", float(f1))

    # -------------------------------
    # Log model artifact
    # -------------------------------
    mlflow.sklearn.log_model(model, "fraud_xgboost_model")

print("XGBoost training complete.")
