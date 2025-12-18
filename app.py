import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# =============================
# Load Model & Preprocessor
# =============================
@st.cache_resource

def load_artifacts():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    model = load_model("best_ann_model.h5")
    return preprocessor, model

preprocessor, model = load_artifacts()

# =============================
# Training Columns (VERY IMPORTANT)
# =============================
TRAINING_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

# =============================
# Sidebar - Developer Info
# =============================
st.sidebar.title("👨‍💻 Developer Information")
st.sidebar.markdown("""
**Name:** Pranav Gaikwad  
**Role:** Aspiring AI / ML Engineer  
**Contact** 7028719844

📧 **Email:** gaikwadpranav988@gmail.com  
🔗 **LinkedIn:** https://www.linkedin.com/in/pranav-gaikwad-0b94032a  
💻 **GitHub:** https://github.com/pranavgaikwad51
""")

st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Project:** Telecom Customer Churn Prediction")
st.sidebar.markdown("🧠 **Model:** Deep Learning (ANN)")
st.sidebar.markdown("⚙️ **Stack:** TensorFlow | Scikit-learn | Streamlit")

# =============================
# Main Title
# =============================
st.title("📡 Telecom Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to **churn (leave)** or **stay** based on customer and service details.")

# =============================
# User Inputs (Clean UI)
# =============================
st.subheader("📝 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])

with col2:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])

with col3:
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    payment_method = st.selectbox(
        "Payment Method",
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    )
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])

# =============================
# Build Full Feature Vector
# =============================
input_data = pd.DataFrame(columns=TRAINING_COLUMNS)

input_data.loc[0] = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': internet_service,
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': tenure * monthly_charges
}

# =============================
# Prediction
# =============================
st.markdown("---")

if st.button("🔍 Predict Churn"):
    try:
        processed_input = preprocessor.transform(input_data)
        prediction = model.predict(processed_input)[0][0]  # 0–1

        churn_percent = prediction * 100
        stay_percent = (1 - prediction) * 100

        st.subheader("📊 Prediction Result")

        if prediction >= 0.5:
            st.error(
                f"❌ Customer is **likely to churn** (Probability: {churn_percent:.0f}%)"
            )
        else:
            st.success(
                f"✅ Customer is **likely to stay** (Probability: {stay_percent:.0f}%)"
            )

    except Exception as e:
        st.error("⚠️ Prediction failed.")
        st.text(str(e))

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Deep Learning")
