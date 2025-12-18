# Telecom Customer Churn Prediction using Deep Learning

## Overview
Customer churn is one of the most critical challenges faced by telecom companies. Acquiring a new customer is significantly more expensive than retaining an existing one. This project focuses on building an end-to-end **Telecom Customer Churn Prediction system** using **Deep Learning (Artificial Neural Network)** and deploying it as an interactive **Streamlit web application**.

The application predicts whether a customer is likely to **churn (leave the service)** or **stay**, based on demographic details, account information, and subscribed telecom services.

---

## Problem Statement
Telecom companies lose substantial revenue when customers discontinue their services. Traditional reactive approaches fail to identify at-risk customers early. The challenge is to build a predictive model that can accurately identify customers who are likely to churn so that proactive retention strategies can be applied.

---

## Objective
- To analyze telecom customer data and identify churn patterns  
- To build a **binary classification model** using Deep Learning (ANN)  
- To preprocess mixed data types using a robust preprocessing pipeline  
- To deploy the trained model using **Streamlit** for real-time predictions  
- To create a clean, user-friendly interface suitable for business users  

---

## Dataset
- **Source:** Kaggle – Telecom Customer Churn Dataset  
- **Link:** https://www.kaggle.com/datasets/mosapabdelghany/telcom-customer-churn-dataset  

### Dataset Description
Each row represents a single telecom customer with details such as:
- Demographics (gender, senior citizen, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (internet service, online security, streaming, tech support)
- Billing information (monthly charges, total charges)

**Target Variable:**
- `Churn` → Yes / No (mapped to 1 / 0)

---

## Tools & Libraries
- **Python**
- **Pandas & NumPy** – Data manipulation
- **Scikit-learn** – Preprocessing & feature engineering
- **TensorFlow / Keras** – Deep Learning (ANN model)
- **Streamlit** – Web application deployment
- **Pickle** – Saving preprocessing pipeline

---

## Model Architecture
The model is an **Artificial Neural Network (ANN)** designed for binary classification:
- Input layer based on preprocessed feature vector  
- Hidden Dense layers with ReLU activation  
- Dropout layers for regularization  
- Output layer with Sigmoid activation  

The model outputs a **probability score** indicating the likelihood of churn.

---

## Data Preprocessing
- Removed identifier column (`customerID`)
- Encoded categorical variables using **OneHotEncoder**
- Scaled numerical features using **StandardScaler**
- Used **ColumnTransformer** to combine all preprocessing steps
- Saved the preprocessing pipeline as `preprocessor.pkl`

The same preprocessing pipeline is reused during deployment to ensure consistency.

---

## Evaluation Metrics
The model performance was evaluated using:
- Accuracy  
- Binary Cross-Entropy Loss  

Probability-based predictions are used in the deployed app to present results in percentage form for better interpretability.

---

## Results
- The ANN model successfully learned churn patterns from customer behavior
- The deployed application predicts churn probability in real time
- Results are displayed in a user-friendly format (e.g., **70% chance to stay**)

This project demonstrates how Deep Learning can be applied to solve real-world business problems effectively.

---

## Streamlit Application

### Features
- Interactive input form for customer details
- Advanced service options hidden under expandable section
- Real-time churn prediction
- Probability displayed in percentage format
- Professional sidebar with developer information

### How to Run the App
```bash
pip install -r requirements.txt
streamlit run app.py

Project Structure
telecom-churn-prediction/
│
├── app.py
├── best_ann_model.h5
├── preprocessor.pkl
├── requirements.txt
└── README.md
Acknowledgements

Kaggle for providing the dataset

Streamlit community for deployment resources

TensorFlow and Scikit-learn documentation

Author

Pranav Gaikwad
Aspiring AI / ML Engineer

📧 Email: gaikwadpranav988@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/pranav-gaikwad-0b94032a

💻 GitHub: https://github.com/pranavgaikwad51
License

This project is licensed for educational and learning purposes.
Feel free to fork and modify with proper attribution.
