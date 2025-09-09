# app.py

import streamlit as st
import numpy as np
import joblib
import time

# --- Page Config ---
st.set_page_config(page_title="Executive Risk Analysis", page_icon="💼", layout="wide")

# --- Load Model ---
try:
    model = joblib.load('credit_risk_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Ensure 'credit_risk_model.pkl' is in the directory.")
    st.stop()

# --- Custom CSS for styled boxes ---
st.markdown("""
<style>
.metric-box {
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    margin: 5px;
    background-color: #073b4c; /* secondaryBackgroundColor from theme */
}
.metric-box h3 {
    color: #fca311; /* primaryColor from theme */
    margin-bottom: 5px;
}
.metric-box p {
    font-size: 1.5rem;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# --- App UI ---
st.title("💼 Executive Credit Risk Dashboard")

# --- Input Fields using columns for a compact layout ---
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income ($)", 10000, 1000000, 50000, 1000)
    with c2:
        home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
        loan_amnt = st.number_input("Loan Amount ($)", 1000, 100000, 10000, 500)
    with c3:
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        emp_length = st.slider("Employment Length (years)", 0, 40, 5)

# Hidden inputs in an expander for less critical info
with st.expander("Additional Details"):
    c4, c5, c6 = st.columns(3)
    with c4:
        loan_intent = st.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "MEDICAL", "VENTURE", "PERSONAL", "EDUCATION", "HOMEIMPROVEMENT"])
    with c5:
        loan_int_rate = st.number_input("Interest Rate (%)", 5.0, 40.0, 10.0)
    with c6:
        default_on_file = st.radio("Previous Default?", ("No", "Yes"), horizontal=True)
        cred_hist_length = st.number_input("Credit History Length (years)", 1, 50, 8)

if st.button("Run Full Spectrum Analysis", type="primary", use_container_width=True):
    # --- Data Processing ---
    loan_percent_income = loan_amnt / income if income > 0 else 0.0
    home_map = {"RENT": 3, "OWN": 2, "MORTGAGE": 1, "OTHER": 0}
    intent_map = {"EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2, "PERSONAL": 3, "DEBTCONSOLIDATION": 4, "HOMEIMPROVEMENT": 5}
    grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    default_map = {"Yes": 1, "No": 0}

    input_data = np.array([[
        age, income, home_map[home_ownership], emp_length, intent_map[loan_intent],
        grade_map[loan_grade], loan_amnt, loan_int_rate, loan_percent_income,
        default_map[default_on_file], cred_hist_length
    ]])

    # --- Simulated Long Process with st.status ---
    with st.status("Performing deep analysis...", expanded=True) as status:
        st.write("Connecting to credit bureau...")
        time.sleep(1.5)
        st.write("Analyzing financial history...")
        time.sleep(2)
        st.write("Running predictive models...")
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        time.sleep(1.5)
        status.update(label="Analysis complete!", state="complete", expanded=False)

    st.header("Analysis Report")
    
    # --- Display Results in Styled Boxes ---
    res_c1, res_c2, res_c3 = st.columns(3)
    
    with res_c1:
        st.markdown('<div class="metric-box"><h3>Final Verdict</h3></div>', unsafe_allow_html=True)
        if prediction[0] == 0:
            st.success("🟢 LOW RISK")
        else:
            st.error("🔴 HIGH RISK")
    
    with res_c2:
        st.markdown(f'<div class="metric-box"><h3>High Risk Probability</h3><p>{prediction_proba[0][1]:.2%}</p></div>', unsafe_allow_html=True)
        
    with res_c3:
        st.markdown(f'<div class="metric-box"><h3>Loan-to-Income</h3><p>{loan_percent_income:.2%}</p></div>', unsafe_allow_html=True)
