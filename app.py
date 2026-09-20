"""Customer Churn Prediction System"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

BLUE, RED = "#2E86DE", "#EE5A52"
sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.autolayout": True})
plt.rcParams["font.family"] = "Times New Roman"
CHURN_PALETTE = {0: BLUE, 1: RED}

st.markdown("""
<style>
html, body, .stApp, p, li, label, h1, h2, h3, h4, button, input {
    font-family: "Times New Roman", serif !important;
}
.hero {background: linear-gradient(90deg, #1F4E9C, #2E86DE); padding: 1.6rem 2rem;
       border-radius: 14px; color: white; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0; font-size: 2.1rem;}
.hero p {margin: .3rem 0 0; opacity: .9;}
div[data-testid="stMetric"] {background: #F1F5FB; border: 1px solid #DCE6F5;
       padding: 1rem 1.2rem; border-radius: 12px;}
.stTabs [data-baseweb="tab"] {font-weight: 600; font-size: 1rem;}
.footer {text-align: center; color: #7A869A; font-size: .85rem; margin-top: 2.5rem;
         padding-top: 1rem; border-top: 1px solid #E3E9F2;}
</style>
""", unsafe_allow_html=True)


# ---------- 1. Data collection ----------
def make_synthetic(n=3000, seed=42):
    """Fallback dataset (Telco-style) so the app runs without a CSV."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.35, .45, .2])
    monthly = np.round(20 + 30 * (internet == "Fiber optic") + 15 * (internet == "DSL")
                       + rng.normal(0, 8, n), 2).clip(18)
    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[.84, .16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[.3, .7]),
        "tenure": tenure,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer", "Credit card"], n),
        "MonthlyCharges": monthly,
    })
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2).astype(str)
    logit = (-0.8 + 1.4 * (contract == "Month-to-month") - 0.03 * tenure
             + 0.02 * (monthly - 60) + 0.5 * (df["PaymentMethod"] == "Electronic check"))
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


DEFAULT_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    if os.path.exists(DEFAULT_CSV):
        return pd.read_csv(DEFAULT_CSV)
    return make_synthetic()


# ---------- 2. Preprocessing + 4. Feature engineering ----------
def preprocess(df):
    df = df.drop_duplicates().copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df = df.drop(columns=[c for c in ["customerID"] if c in df.columns])  # irrelevant
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(df["Churn"]).astype(int)
    # new features
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["ContractGroup"] = df["Contract"].map(
        {"Month-to-month": "Short", "One year": "Long", "Two year": "Long"})
    return df


def encode(df):
    X = pd.get_dummies(df.drop(columns="Churn"), drop_first=True)
    return X.astype(float), df["Churn"]


@st.cache_resource
def train_models(df):
    X, y = encode(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(Xtr_s, ytr)
        pred = m.predict(Xte_s)
        fitted[name] = (m, confusion_matrix(yte, pred))
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(yte, pred),
                     "Precision": precision_score(yte, pred, zero_division=0),
                     "Recall": recall_score(yte, pred),
                     "F1": f1_score(yte, pred)})
    return pd.DataFrame(rows).set_index("Model"), fitted, scaler, list(X.columns)


# ---------- UI ----------
st.markdown("""
<div class="hero">
  <h1><center>Customer Churn Prediction System</center></h1>
  <p><center>Find customers at risk of leaving, compare ML models, and predict churn in real time.</center></p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    up = st.file_uploader("Upload churn CSV (Telco format)", type="csv")
    st.info("No file uploaded? A synthetic Telco-style dataset is used." if up is None
            else "Using your uploaded dataset.")

raw = load_data(up)
df = preprocess(raw)
tab_eda, tab_model, tab_pred = st.tabs(["EDA", "Model comparison", "Predict a customer"])

with tab_eda:
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['Churn'].mean():.1%}")
    c3.metric("Avg monthly charge", f"${df['MonthlyCharges'].mean():.2f}")

    a, b = st.columns(2)
    fig, ax = plt.subplots()
    counts = df["Churn"].map({0: "Stayed", 1: "Left"}).value_counts().reindex(["Stayed", "Left"])
    counts.plot.pie(autopct="%1.1f%%", ax=ax, ylabel="", colors=[BLUE, RED],
                    wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Churn distribution")
    a.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(data=df, x="MonthlyCharges", hue="Churn", kde=True, palette=CHURN_PALETTE, ax=ax)
    ax.set_title("Monthly charges vs churn")
    b.pyplot(fig)

    a, b = st.columns(2)
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", palette=CHURN_PALETTE, ax=ax)
    ax.set_title("Tenure vs churn")
    a.pyplot(fig)

    fig, ax = plt.subplots()
    df.groupby("Contract")["Churn"].mean().plot.bar(ax=ax, color=RED)
    ax.set_title("Churn rate by contract type")
    ax.set_ylabel("Churn rate")
    b.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f",
                cmap="Blues", ax=ax)
    ax.set_title("Correlation heatmap")
    st.pyplot(fig)

results, fitted, scaler, cols = train_models(df)

with tab_model:
    st.subheader("Evaluation on held-out test set (20%)")
    st.dataframe(results.style.format("{:.3f}").highlight_max(axis=0, color="#c6efce"))
    pick = st.selectbox("Confusion matrix for", list(fitted))
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(fitted[pick][1], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    if pick == "Random Forest":
        imp = pd.Series(fitted[pick][0].feature_importances_, index=cols).nlargest(10)
        fig, ax = plt.subplots()
        imp[::-1].plot.barh(ax=ax, color=BLUE)
        ax.set_title("Top 10 feature importances")
        st.pyplot(fig)

with tab_pred:
    st.subheader("Will this customer churn?")
    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Tenure (months)", 1, 72, 12)
    monthly = c2.slider("Monthly charges", 18.0, 120.0, 70.0)
    contract = c3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = c1.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment = c2.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer", "Credit card"])
    senior = c3.selectbox("Senior citizen", [0, 1])

    if st.button("Predict"):
        # start from the most common profile, then override the chosen inputs
        base = df.drop(columns="Churn").mode().iloc[0].copy()
        base.update({"tenure": tenure, "MonthlyCharges": monthly, "Contract": contract,
                     "InternetService": internet, "PaymentMethod": payment,
                     "SeniorCitizen": senior, "TotalCharges": monthly * tenure,
                     "AvgMonthlySpend": monthly,
                     "ContractGroup": "Short" if contract == "Month-to-month" else "Long"})
        row = pd.get_dummies(pd.DataFrame([base])).reindex(columns=cols, fill_value=0).astype(float)
        prob = fitted["Random Forest"][0].predict_proba(scaler.transform(row))[0, 1]
        st.metric("Churn probability (Random Forest)", f"{prob:.1%}")
        (st.error if prob >= 0.5 else st.success)(
            "⚠️ Likely to churn - consider a retention offer." if prob >= 0.5
            else "✅ Likely to stay.")


st.markdown('<div class="footer">Customer Churn Prediction System</div>', unsafe_allow_html=True)
