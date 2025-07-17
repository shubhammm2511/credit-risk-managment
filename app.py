import streamlit as st
import numpy as np
import joblib
import os

# Page settings
st.set_page_config(
    page_title="Credit Risk Classifier",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==== Sidebar ====
with st.sidebar:
    st.markdown("## 📊 Credit Risk Classification")
    st.markdown("This app predicts whether a loan applicant is **Low Risk** or **High Risk** based on their financial details.")
    st.markdown("---")
    st.markdown("**👨‍💻 Made by:** Shubham Tiwary")
    st.markdown("[📧 Contact me](mailto:shubhamtiwari2511@outlook.com)")

# ==== Main Title ====
st.title("💳 Credit Risk Classifier")
st.markdown("Enter the applicant's financial details to assess their credit risk.")
st.divider()

# ==== Inputs ====
st.subheader("📋 Applicant Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100)
    income = st.number_input("Annual Income (₹)", 10000)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    emp_length = st.slider("Employment Length (years)", 0, 20)
    default_on_file = st.selectbox("Default on File", ["Y", "N"])

with col2:
    loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount (₹)", 1000)
    loan_int_rate = st.number_input("Interest Rate (%)", 5.0, 40.0)
    cred_hist_length = st.slider("Credit History Length (years)", 1, 50)

loan_percent_income = loan_amnt / income if income > 0 else 0.0

# ==== Encode Input ====
home_map = {"RENT": 3, "OWN": 2, "MORTGAGE": 1, "OTHER": 0}
intent_map = {"EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2, "PERSONAL": 3, "DEBTCONSOLIDATION": 4, "HOMEIMPROVEMENT": 5}
grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
default_map = {"Y": 1, "N": 0}

input_data = np.array([[ 
    age,
    income,
    home_map[home_ownership],
    emp_length,
    intent_map[loan_intent],
    grade_map[loan_grade],
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    default_map[default_on_file],
    cred_hist_length
]])

# ==== Load model and predict ====
if os.path.exists("credit_risk_model.pkl"):
    model = joblib.load("credit_risk_model.pkl")
else:
    st.error("🚨 Model file not found.")
    st.stop()

if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    result = "🟢 Low Risk" if prediction[0] == 0 else "🔴 High Risk"
    st.success(f"**Prediction:** {result}")
