# Customer Churn Prediction System
Sqrock IT Solutions - Data Science Internship, Project Phase 1, Task 4.

Predicts whether a telecom customer will churn using Logistic Regression, Decision Tree,
Random Forest and KNN. Includes EDA, model comparison (Accuracy, Precision, Recall, F1,
confusion matrix) and a live single-customer prediction form.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Dataset
Upload the Kaggle "Telco Customer Churn" CSV in the sidebar. Without an upload, a synthetic
Telco-style dataset is used.

## Deploy
Push to GitHub, then create an app at share.streamlit.io pointing to `app.py`.

## Live Link
https://customer-churn-prediction-or3aljsqhfosy4vav9tzsy.streamlit.app/
