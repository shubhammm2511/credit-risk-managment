import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('credit_risk_model.pkl')

st.title("💳 Credit Risk Classification")
st.write("Enter details to predict whether the applicant is Low Risk or High Risk.")

# Input fields
age = st.number_input("Age", 18, 100)
income = st.number_input("Annual Income", 10000)
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
emp_length = st.slider("Employment Length (years)", 0, 20)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount", 1000)
loan_int_rate = st.number_input("Interest Rate (%)", 5.0, 40.0)
loan_percent_income = loan_amnt / income if income > 0 else 0.0
default_on_file = st.selectbox("Default on File", ["Y", "N"])
cred_hist_length = st.slider("Credit History Length (years)", 1, 50)

# Encode inputs
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

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.subheader(f"Prediction: {'🔵 Low Risk' if prediction[0] == 0 else '🔴 High Risk'}")
