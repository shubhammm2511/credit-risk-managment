import streamlit as st
import numpy as np
import joblib
import time

# --- Load Model ---
try:
    model = joblib.load('credit_risk_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Ensure 'credit_risk_model.pkl' is in the directory.")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="Customizable Risk App", page_icon="🎨", layout="centered")

# --- Initialize Session State ---
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "low_risk_color" not in st.session_state:
    st.session_state.low_risk_color = "#28a745"
if "high_risk_color" not in st.session_state:
    st.session_state.high_risk_color = "#dc3545"

# --- Dynamic CSS based on session state ---
def get_theme_css(theme, low_color, high_color):
    if theme == "dark":
        return f"""
        <style>
            .stApp {{
                background-color: #0E1117;
                color: #FAFAFA;
            }}
            .stProgress > div > div > div > div {{
                background-color: {low_color};
            }}
            .low-risk-text {{ color: {low_color}; }}
            .high-risk-text {{ color: {high_color}; }}
        </style>
        """
    else:
        return f"""
        <style>
            .stApp {{
                background-color: #FFFFFF;
                color: #0E1117;
            }}
            .stProgress > div > div > div > div {{
                background-color: {low_color};
            }}
            .low-risk-text {{ color: {low_color}; }}
            .high-risk-text {{ color: {high_color}; }}
        </style>
        """

# Apply the dynamic theme
st.markdown(get_theme_css(st.session_state.theme, st.session_state.low_risk_color, st.session_state.high_risk_color), unsafe_allow_html=True)

# --- Sidebar for Customization ---
with st.sidebar:
    st.header("🎨 Appearance Settings")
    
    # Theme Toggle
    if st.toggle("Enable Dark Mode", value=(st.session_state.theme == "dark")):
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

    # Color Pickers
    st.session_state.low_risk_color = st.color_picker("Low Risk Color", st.session_state.low_risk_color)
    st.session_state.high_risk_color = st.color_picker("High Risk Color", st.session_state.high_risk_color)
    
    # Rerun to apply changes immediately
    st.button("Apply Changes")

# --- Main App UI ---
st.title("Customizable Credit Risk Analyzer")
st.info("Use the sidebar to change the theme and colors!")

# Input fields
with st.container(border=True):
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual Income", 10000, 200000, 50000)
    loan_amnt = st.number_input("Loan Amount", 1000, 50000, 10000)

# Dummy inputs for demonstration
home_ownership, emp_length, loan_intent, loan_grade, loan_int_rate, default_on_file, cred_hist_length = "MORTGAGE", 5, "PERSONAL", "B", 10.0, "No", 8

if st.button("Analyze Risk", use_container_width=True):
    # --- Data Processing ---
    # In a real app, you would collect all inputs as before
    loan_percent_income = loan_amnt / income if income > 0 else 0.0
    home_map, intent_map, grade_map, default_map = {"MORTGAGE": 1}, {"PERSONAL": 3}, {"B": 1}, {"No": 0}
    
    input_data = np.array([[
        age, income, home_map[home_ownership], emp_length, intent_map[loan_intent],
        grade_map[loan_grade], loan_amnt, loan_int_rate, loan_percent_income,
        default_map[default_on_file], cred_hist_length
    ]])

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    high_risk_prob = prediction_proba[0][1]

    # --- Animated Progress Bar ---
    st.subheader("Final Assessment")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for percent_complete in range(101):
        time.sleep(0.02)
        progress_bar.progress(percent_complete)
        if percent_complete < 50:
            progress_text.text(f"Processing... {percent_complete}%")
        else:
            progress_text.text(f"Finalizing... {percent_complete}%")

    progress_text.empty()
    progress_bar.empty()

    # --- Display Results with Custom Colors ---
    if prediction[0] == 0:
        st.markdown(f'## <span class="low-risk-text">Result: LOW RISK</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'## <span class="high-risk-text">Result: HIGH RISK</span>', unsafe_allow_html=True)
    
    st.write(f"The probability of the applicant being high-risk is **{high_risk_prob:.2%}**.")
