import streamlit as st
import numpy as np
import joblib
import time

# Load trained model
try:
    model = joblib.load('credit_risk_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'credit_risk_model.pkl' is in the directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💳",
    layout="centered"
)


# --- UI ---
st.title("🗂️ Advanced Credit Risk Assessment")
st.info("Complete the sections below and click 'Predict' to assess the credit risk.")

# --- Input Fields inside Expanders ---
with st.expander("👤 Personal & Employment Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
    with col2:
        income = st.number_input("Annual Income ($)", 10000, 1000000, 50000, 1000)
        emp_length = st.slider("Employment Length (years)", 0, 40, 5)

with st.expander("💰 Loan & Credit Details", expanded=True):
    col3, col4 = st.columns(2)
    with col3:
        loan_intent = st.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "MEDICAL", "VENTURE", "PERSONAL", "EDUCATION", "HOMEIMPROVEMENT"])
        loan_amnt = st.number_input("Loan Amount ($)", 1000, 100000, 10000, 500)
        default_on_file = st.radio("Previous Default?", ("No", "Yes"), horizontal=True)
    with col4:
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_int_rate = st.number_input("Interest Rate (%)", 5.0, 40.0, 10.0)
        cred_hist_length = st.slider("Credit History Length (years)", 1, 50, 8)


# --- Prediction Button ---
if st.button("Assess Risk", type="primary", use_container_width=True):
    
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
    
    # --- Prediction ---
    with st.spinner('Performing detailed analysis...'):
        time.sleep(1)
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
    st.subheader("📋 Assessment Summary")

    if prediction[0] == 0:
        st.success("Result: LOW RISK APPLICANT")
        st.balloons()
        st.markdown(f"""
        Based on the provided data, the applicant has a low probability of default (**{prediction_proba[0][1]:.1%}**).
        
        **Key Positive Indicators:**
        - A stable financial profile is indicated.
        - The credit history appears to be in good standing.
        
        **Recommendation:** This applicant is a good candidate for loan approval.
        """)
    else:
        st.error("Result: HIGH RISK APPLICANT")
        st.markdown(f"""
        The model has identified this applicant as high-risk, with a default probability of **{prediction_proba[0][1]:.1%}**.
        
        **Potential Risk Factors:**
        - The loan amount might be high relative to income.
        - Previous credit history or a high interest rate could be contributing factors.
        
        **Recommendation:** Further review is strongly recommended before approving this loan. Consider requesting additional documentation or offering a smaller loan amount.
        """)
