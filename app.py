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
st.set_page_config(page_title="Guided Credit Application", page_icon="📝", layout="centered")

st.title("📝 Guided Credit Risk Application")
st.write("Please complete the following sections to receive a risk assessment.")

# --- Tabbed Interface ---
tab1, tab2, tab3 = st.tabs(["Step 1: Personal Info", "Step 2: Loan Details", "Step 3: Get Assessment"])

with tab1:
    st.header("👤 Personal & Employment Details")
    st.session_state.age = st.number_input("Age", 18, 100, 30)
    st.session_state.income = st.number_input("Annual Income ($)", 10000, 1000000, 50000, 1000)
    st.session_state.home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
    st.session_state.emp_length = st.slider("Employment Length (years)", 0, 40, 5)
    st.info("Please proceed to Step 2.")

with tab2:
    st.header("💰 Loan & Credit History")
    st.session_state.loan_intent = st.selectbox("What is the purpose of the loan?", ["DEBTCONSOLIDATION", "MEDICAL", "VENTURE", "PERSONAL", "EDUCATION", "HOMEIMPROVEMENT"])
    st.session_state.loan_amnt = st.number_input("How much would you like to borrow ($)?", 1000, 100000, 10000, 500)
    st.session_state.loan_grade = st.select_slider("Select a desired loan grade:", ["A", "B", "C", "D", "E", "F", "G"])
    st.session_state.loan_int_rate = st.number_input("Preferred Interest Rate (%)", 5.0, 40.0, 10.0)
    st.session_state.default_on_file = st.radio("Have you defaulted on a loan before?", ("No", "Yes"), horizontal=True)
    st.session_state.cred_hist_length = st.slider("How long is your credit history (years)?", 1, 50, 8)
    st.info("Once you are ready, proceed to Step 3 for the assessment.")

with tab3:
    st.header("📈 Your Risk Assessment")
    
    if st.button("✨ Click Here to Assess Risk", type="primary", use_container_width=True):
        # Check if all keys are in session_state
        required_keys = ['age', 'income', 'home_ownership', 'emp_length', 'loan_intent', 'loan_amnt', 'loan_grade', 'loan_int_rate', 'default_on_file', 'cred_hist_length']
        if not all(key in st.session_state for key in required_keys):
            st.error("Please complete all fields in Step 1 and Step 2 before assessing.")
        else:
            # --- Data Processing and Prediction ---
            loan_percent_income = st.session_state.loan_amnt / st.session_state.income if st.session_state.income > 0 else 0.0
            
            home_map = {"RENT": 3, "OWN": 2, "MORTGAGE": 1, "OTHER": 0}
            intent_map = {"EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2, "PERSONAL": 3, "DEBTCONSOLIDATION": 4, "HOMEIMPROVEMENT": 5}
            grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
            default_map = {"Yes": 1, "No": 0}

            input_data = np.array([[
                st.session_state.age, st.session_state.income, home_map[st.session_state.home_ownership],
                st.session_state.emp_length, intent_map[st.session_state.loan_intent], grade_map[st.session_state.loan_grade],
                st.session_state.loan_amnt, st.session_state.loan_int_rate, loan_percent_income,
                default_map[st.session_state.default_on_file], st.session_state.cred_hist_length
            ]])

            with st.spinner('Analyzing your profile...'):
                time.sleep(1)
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)

            if prediction[0] == 0:
                st.success(f"**Result: LOW RISK** (Probability: {prediction_proba[0][0]:.2%})")
                st.balloons()
            else:
                st.error(f"**Result: HIGH RISK** (Probability: {prediction_proba[0][1]:.2%})")
            
            st.write("---")
            st.subheader("Summary of Your Application:")
            # Display a summary of the inputs
            summary_data = {
                "Age": st.session_state.age, "Annual Income": f"${st.session_state.income:,}",
                "Loan Amount": f"${st.session_state.loan_amnt:,}", "Loan Intent": st.session_state.loan_intent,
                "Interest Rate": f"{st.session_state.loan_int_rate}%", "Previous Default": st.session_state.default_on_file
            }
            st.json(summary_data)
