import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

# =========================
# Prepare Directory
# =========================

os.makedirs("models", exist_ok=True)

# =========================
# Load Dataset
# =========================

data = pd.read_csv("data/creditcard.csv")
X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# Train Model
# =========================

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Threshold Optimization
# =========================

y_probs = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

# =========================
# Save Artifacts
# =========================

joblib.dump(model, "models/fraud_xgboost.pkl")

with open("models/best_threshold.txt", "w") as f:
    f.write(str(best_threshold))

# Save feature statistics for drift detection
feature_means = X_train.mean().values
feature_stds = X_train.std().values

np.save("models/feature_means.npy", feature_means)
np.save("models/feature_stds.npy", feature_stds)

print("Saved model to models/fraud_xgboost.pkl")
print("Saved threshold to models/best_threshold.txt")
print("Saved drift statistics")
print("Best threshold:", best_threshold)