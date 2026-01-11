import streamlit as st
import pandas as pd
import numpy as np
import joblib

cluster_model = joblib.load("output/kmeans_model.pkl")
delivery_model = joblib.load("output/product_delivery_model.pkl")
delivery_scaler = joblib.load("output/scaler.pkl")
delivery_features = joblib.load("output/delivery_model_features.pkl")

st.title("Credit Card Customer Segmentation and Product Delivery Prediction")

uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    cust_id = df["CUST_ID"]
    X = df.drop(columns=["CUST_ID"], errors="ignore")

    if "Cluster" in X.columns:
        X = X.drop(columns=["Cluster"])

    log_columns = [
        'BALANCE','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES',
        'CASH_ADVANCE','CASH_ADVANCE_TRX','PURCHASES_TRX',
        'PAYMENTS','MINIMUM_PAYMENTS'
    ]
# Apply log transform safely
    for col in log_columns:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = np.log1p(X[col].fillna(0))

# Ensure all required model features exist
    for col in delivery_features:
        if col not in X.columns:
            X[col] = 0

# Keep only model features in correct order
    X = X[delivery_features]

# Replace any remaining NaNs
    X = X.fillna(0)

# Scale
    X_scaled = delivery_scaler.transform(X)
    
    cluster_preds = cluster_model.predict(X_scaled)
    delivery_preds = delivery_model.predict(X_scaled)

    result = df.copy()
    result["Cluster"] = cluster_preds
    result["Product_Delivery_Prediction"] = delivery_preds

    st.success("Prediction Completed Successfully")
    st.dataframe(result)

