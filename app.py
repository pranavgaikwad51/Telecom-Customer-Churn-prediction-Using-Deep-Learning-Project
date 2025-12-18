import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# -----------------------------
# Load Preprocessor & Model
# -----------------------------
@st.cache_resource

def load_artifacts():
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    model = load_model('best_ann_model.h5')
    return preprocessor, model

preprocessor, model = load_artifacts()

# -----------------------------
# Sidebar - Developer Info
# -----------------------------
st.sidebar.title("👨‍💻 Developer Information")
st.sidebar.markdown("""
**Name:** Pranav Gaikwad  
**Role:** Aspiring AI / ML Engineer  
 

📧 **Email:** gaikwadpranav988@gmail.com  
🔗 **LinkedIn:** [Profile](https://www.linkedin.com/in/pranav-gaikwad-0b94032a)  
💻 **GitHub:** [Repository](https://github.com/pranavgaikwad51)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Project:** Telecom Customer Churn Prediction")
st.sidebar.markdown("🧠 **Model:** Deep Learning (ANN)")
st.sidebar.markdown("⚙️ **Framework:** TensorFlow & Streamlit")

# -----------------------------
# Main Title
# -----------------------------
st.title("📡 Telecom Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to **churn (leave)** or **stay** based on service and account details.")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("📝 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

with col2:
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])

with col3:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ['Yes', 'No'])

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'tenure': [tenure],
    'InternetService': [internet_service],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges]
})

# -----------------------------
# Prediction
# -----------------------------
st.markdown("---")
if st.button("🔍 Predict Churn"):
    try:
        processed_input = preprocessor.transform(input_data)
        prediction = model.predict(processed_input)[0][0]

        if prediction >= 0.5:
            st.error(f"❌ Customer is **likely to churn** (Probability: {prediction:.2f})")
        else:
            st.success(f"✅ Customer is **likely to stay** (Probability: {1 - prediction:.2f})")
    except Exception as e:
        st.warning("⚠️ Feature mismatch detected. Ensure input features match training data.")
        st.text(str(e))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Deep Learning")
