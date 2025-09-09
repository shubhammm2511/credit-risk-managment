import streamlit as st
import numpy as np
import joblib
import os

# Load trained model
try:
    model = joblib.load('credit_risk_model.pkl')  
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'credit_risk_model.pkl' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .stTitle {
        color: #007bff;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stHeader, .stSubheader {
        color: #007bff;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Credit Risk Classification")
st.write("Enter details to predict whether the applicant is Low Risk or High Risk.")

# --- Input Fields ---
tab1, tab2 = st.tabs(["Applicant Information", "Loan Details"])

with tab1:
    st.header("Applicant Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100)
        income = st.number_input("Annual Income", 10000)
    with col2:
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        emp_length = st.slider("Employment Length (years)", 0, 20)

with tab2:
    st.header("Loan Details")
    col3, col4 = st.columns(2)
    with col3:
        loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    with col4:
        loan_amnt = st.number_input("Loan Amount", 1000)
        loan_int_rate = st.number_input("Interest Rate (%)", 5.0, 40.0)

# --- Additional Fields ---
st.header("Additional Information")
cred_hist_length = st.slider("Credit History Length (years)", 1, 50)
default_on_file = st.selectbox("Default on File", ["Y", "N"])

loan_percent_income = loan_amnt / income if income > 0 else 0.0

# --- Encoding and Prediction ---
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

if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.success("Prediction: 🔵 Low Risk")
            st.balloons()
        else:
            st.error("Prediction: 🔴 High Risk")
            st.warning("Please review the applicant's details carefully.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.markdown("---")
st.write("Credit Risk Classification Model")
