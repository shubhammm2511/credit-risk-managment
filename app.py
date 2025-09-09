import streamlit as st
import numpy as np
import joblib
import shap  # pip install shap
import pandas as pd
import matplotlib.pyplot as plt

# --- Model & Explainer Loading ---
# This assumes your 'credit_risk_model.pkl' is a tree-based model
try:
    model = joblib.load('credit_risk_model.pkl')
    # Create a SHAP explainer object
    explainer = shap.TreeExplainer(model)
except (FileNotFoundError, Exception) as e:
    st.error(f"Model or SHAP explainer could not be loaded. Error: {e}")
    st.info("This option requires a tree-based model (e.g., RandomForest) and the 'shap' library.")
    st.stop()


# --- Page Configuration ---
st.set_page_config(page_title="Explainable Credit AI", page_icon="🧠", layout="wide")

st.title("🧠 Explainable Credit Risk AI")
st.write("This tool not only predicts credit risk but also shows you *why* a decision was made.")

# --- Feature Names (must match model training order) ---
feature_names = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
                 'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                 'cb_person_default_on_file', 'cb_person_cred_hist_length']

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Applicant Details")
    age = st.number_input("Age", 18, 100, 25)
    income = st.number_input("Annual Income", 10000, 200000, 65000)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    emp_length = st.slider("Employment Length (years)", 0, 20, 4)
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount", 1000, 50000, 12000)
    loan_int_rate = st.slider("Interest Rate (%)", 5.0, 40.0, 11.5)
    default_on_file = st.selectbox("Default on File", ["N", "Y"])
    cred_hist_length = st.slider("Credit History Length (years)", 1, 50, 3)

# --- Data Processing ---
home_map = {"RENT": 3, "OWN": 2, "MORTGAGE": 1, "OTHER": 0}
intent_map = {"EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2, "PERSONAL": 3, "DEBTCONSOLIDATION": 4, "HOMEIMPROVEMENT": 5}
grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
default_map = {"Y": 1, "N": 0}

loan_percent_income = loan_amnt / income if income > 0 else 0.0

input_data = np.array([
    age, income, home_map[home_ownership], emp_length, intent_map[loan_intent],
    grade_map[loan_grade], loan_amnt, loan_int_rate, loan_percent_income,
    default_map[default_on_file], cred_hist_length
])

# Create a DataFrame for SHAP which requires feature names
input_df = pd.DataFrame([input_data], columns=feature_names)

# --- Prediction and Explanation ---
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]
shap_values = explainer.shap_values(input_df)

# We are interested in the explanation for the "High Risk" class (class 1)
shap_values_class1 = shap_values[1][0]

# --- Display Results ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Outcome")
    if prediction == 0:
        st.success(f"**LOW RISK** (Probability: {prediction_proba[0]:.2%})")
    else:
        st.error(f"**HIGH RISK** (Probability: {prediction_proba[1]:.2%})")
    
    st.write("The SHAP Force Plot below shows the features pushing the prediction.")
    # Visualize the SHAP explanation
    fig = shap.force_plot(
        explainer.expected_value[1],
        shap_values_class1,
        input_df.iloc[0],
        matplotlib=False
    )
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.components.v1.html(fig.html(), height=150)

with col2:
    st.subheader("Key Contributing Factors")
    
    # Create a DataFrame of feature values and their SHAP importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'feature_value': input_df.iloc[0].values,
        'shap_value': shap_values_class1
    })
    feature_importance['abs_shap'] = feature_importance['shap_value'].abs()
    feature_importance = feature_importance.sort_values(by='abs_shap', ascending=False).head(5)

    st.write("Top 5 factors influencing this prediction:")
    for index, row in feature_importance.iterrows():
        if row['shap_value'] > 0:
            st.markdown(f"- **{row['feature'].replace('_', ' ').title()}**: `{row['feature_value']}` <span style='color:red;'>increased</span> risk.", unsafe_allow_html=True)
        else:
            st.markdown(f"- **{row['feature'].replace('_', ' ').title()}**: `{row['feature_value']}` <span style='color:green;'>decreased</span> risk.", unsafe_allow_html=True)
