# 💳 Credit Risk Classification Web App

This project is a **Machine Learning web application** that predicts whether a loan applicant is **Low Risk** or **High Risk** based on their financial details.  
The app is built using **Python, Streamlit, and Machine Learning algorithms** and deployed online for real-time predictions.

---

## 🚀 Live Demo
👉 [Click here to try the app](https://credit-risk-managment-mc7bmqmagej2xnxwyeaa7g.streamlit.app/)  



---

## 📌 Problem Statement
Banks and financial institutions face challenges in identifying high-risk loan applicants.  
This project aims to build a **credit risk prediction model** that helps lenders make **data-driven loan approval decisions**.

---

## 🛠️ Tech Stack
- **Programming Languages:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, XGBoost, Joblib  
- **Frontend:** Streamlit  
- **Deployment:** Streamlit Cloud / Render / Docker  

---

## 📊 Dataset
- **Source:** Public credit dataset (CSV)  
- **Rows:** ~32,000  
- **Features:** Age, Income, Home Ownership, Employment Length, Loan Intent, Loan Grade, Loan Amount, Interest Rate, Default on File, Credit History Length, etc.  
- **Target:** Loan Risk (Low Risk = 0, High Risk = 1)  

---

## 🔬 Methodology
1. **Data Preprocessing**
   - Handled missing values  
   - Encoded categorical features  
   - Normalized numerical values  

2. **Model Training**
   - Logistic Regression  
   - Random Forest  
   - XGBoost  

3. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score  
   - Confusion Matrix  

4. **Deployment**
   - Best model (Random Forest) saved as `.pkl` file  
   - Integrated with **Streamlit UI**  
   - Deployed on cloud  

---

## 📈 Model Performance
| Model                    | Accuracy      | Precision | Recall  | F1-Score |
|-----------------------   |---------------|-----------|-------- |----------|
| Logistic Regression      | ~71.2%        | ~70%      | ~71%    | ~70%     |
| XGBoost                  | ~81.3%        | ~81%      | ~82%    | ~81%     |
| **Random Forest (Best)** | **82.4%** ✅ | **83%**    | **84%** | **83%** |

---

## 🖥️ How to Run Locally
```bash
# Clone the repository
git clone https://github.com/your-username/Credit_risk_classification.git

# Navigate to folder
cd Credit_risk_classification

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
