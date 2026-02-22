import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("🚨 Fraud Detection Dashboard")

st.markdown("### Enter 30 transaction features")

features = []
cols = st.columns(3)

for i in range(30):
    with cols[i % 3]:
        value = st.number_input(f"Feature {i}", value=0.0)
        features.append(value)

if st.button("Predict & Explain"):

    response = requests.post(
        f"{API_URL}/explain",
        json={"features": features}
    )

    result = response.json()

    prob = result["fraud_probability"]

    st.markdown("## 📊 Fraud Probability")
    st.progress(float(prob))

    st.write(f"**Probability:** {prob:.4f}")

    st.markdown("## 🔎 Top Contributing Features")

    explanation = result["top_contributing_features"]

    df = pd.DataFrame(explanation)

    fig, ax = plt.subplots()
    colors = ["red" if v > 0 else "blue" for v in df["impact"]]

    ax.barh(df["feature_name"], df["impact"], color=colors)
    ax.set_xlabel("SHAP Impact")
    ax.set_title("Feature Contribution")

    st.pyplot(fig)