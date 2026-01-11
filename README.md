# Credit Card Customer Segmentation & Product Delivery Prediction App

This project is an end-to-end Machine Learning application that performs customer segmentation using K-Means clustering and predicts product delivery outcomes using a supervised learning model. An interactive Streamlit web app allows users to upload customer data and receive real-time predictions.

---

## Project Goal

Segment credit card customers based on spending and payment behavior to support targeted banking strategies and risk management.

---

## Dataset

Processed credit card transaction dataset containing balances, purchases, payments, and credit limits.

---

## Methodology

1. Data cleaning and missing value handling  
2. Exploratory Data Analysis  
3. Log transformation to reduce skewness  
4. Feature scaling using StandardScaler  
5. PCA for visualization  
6. K-Means clustering with silhouette validation  
7. Cluster profiling  
8. Business feature engineering  
9. Product delivery prediction model training  
10. Web app deployment using Streamlit  

---

## Engineered Business Metrics

- Utilization Ratio  
- Payment Ratio  
- Installment Ratio  
- Cash Advance Intensity  
- Average Purchase Size  

---

## Clustering Outcome

Two customer segments identified:
- Responsible Spenders  
- High Credit Utilizers  

---

## Streamlit Web Application

The Streamlit app allows users to:
- Upload CSV customer data  
- Automatically preprocess input features  
- Predict customer cluster  
- Predict product delivery outcome  
- View results in a table  

---

## Project Structure

credit-card-streamlit-app/  
│  
├ credit_card_app.py  
├ requirements.txt  
├ README.md  
└── output/  
    ├ kmeans_model.pkl  
    ├ product_delivery_model.pkl  
    ├ scaler.pkl  
    └ delivery_model_features.pkl  

---

## Deployment

The application is deployed using Streamlit Community Cloud directly from GitHub.

Steps:
1. Push repository to GitHub  
2. Visit https://share.streamlit.io  
3. Connect GitHub account  
4. Create a new app  
5. Select repository  
6. Set main file path as:

credit_card_app.py

7. Deploy

---

## Input File Requirement

Uploaded CSV must contain:
- CUST_ID column  
- Numerical feature columns used during training  

Missing model features are handled automatically.

---

## Tools Used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit, Joblib

---

## Author

Asit Rochlaney
