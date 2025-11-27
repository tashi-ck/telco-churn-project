import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ------------------------------------------------------
# Page Config
# ------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Telco Customer Churn Prediction App")
st.write("Enter customer details and predict whether they will churn.")

# ------------------------------------------------------
# Load Model + Scaler
# ------------------------------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("../models/churn_model.pkl", "rb"))
    scaler = pickle.load(open("../models/scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# ------------------------------------------------------
# All final columns after feature_engineering.ipynb
# (IMPORTANT: must match df.columns after get_dummies)
# ------------------------------------------------------
final_columns = pickle.load(open("../models/columns.pkl", "rb"))

# ------------------------------------------------------
# User Input Form
# ------------------------------------------------------
st.header("📝 Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", 0, 72)

with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

col3, col4 = st.columns(2)

with col3:
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

with col4:
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0)

# ------------------------------------------------------
# Convert Input to DataFrame
# ------------------------------------------------------
input_dict = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": SeniorCitizen,
    "Partner": 1 if Partner == "Yes" else 0,
    "Dependents": 1 if Dependents == "Yes" else 0,
    "PhoneService": 1 if PhoneService == "Yes" else 0,
    "MultipleLines": 1 if "Yes" in MultipleLines else 0,
    "OnlineSecurity": 1 if "Yes" in OnlineSecurity else 0,
    "OnlineBackup": 1 if "Yes" in OnlineBackup else 0,
    "DeviceProtection": 1 if "Yes" in DeviceProtection else 0,
    "TechSupport": 1 if "Yes" in TechSupport else 0,
    "StreamingTV": 1 if "Yes" in StreamingTV else 0,
    "StreamingMovies": 1 if "Yes" in StreamingMovies else 0,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

# Categorical dummies must be added manually
df_input = pd.DataFrame([input_dict])

# Add dummy columns
df_input = pd.get_dummies(df_input)

# Ensure all missing columns are added
for col in final_columns:
    if col not in df_input.columns:
        df_input[col] = 0

df_input = df_input[final_columns]  # reorder

# Scale numeric columns
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

# ------------------------------------------------------
# Prediction
# ------------------------------------------------------
if st.button("Predict Churn"):
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is **likely to churn** (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is **not likely to churn** (Probability: {probability:.2f})")

