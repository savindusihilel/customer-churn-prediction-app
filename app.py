import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="The Mavericks | Customer Churn Prediction App")

# Load the saved model
model_filename = 'model.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))

# Convert 'Yes'/'No' to 1/0 and other categories to the corresponding numeric format
def preprocess_input(SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService, 
                     MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, 
                     StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure):
    
    # Convert binary categorical variables to numeric
    SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0
    gender_encoded = [1, 0] if gender == "Female" else [0, 1]
    partner_encoded = [1, 0] if Partner == "No" else [0, 1]
    dependents_encoded = [1, 0] if Dependents == "No" else [0, 1]
    phone_service_encoded = [1, 0] if PhoneService == "No" else [0, 1]

    # Convert other categorical variables using one-hot encoding
    multiple_lines_encoded = [0, 0, 0]
    if MultipleLines == "No":
        multiple_lines_encoded[0] = 1
    elif MultipleLines == "No phone service":
        multiple_lines_encoded[1] = 1
    else:
        multiple_lines_encoded[2] = 1

    internet_service_encoded = [0, 0, 0]
    if InternetService == "DSL":
        internet_service_encoded[0] = 1
    elif InternetService == "Fiber optic":
        internet_service_encoded[1] = 1
    else:
        internet_service_encoded[2] = 1

    online_security_encoded = [0, 0, 0]
    if OnlineSecurity == "No":
        online_security_encoded[0] = 1
    elif OnlineSecurity == "No internet service":
        online_security_encoded[1] = 1
    else:
        online_security_encoded[2] = 1

    online_backup_encoded = [0, 0, 0]
    if OnlineBackup == "No":
        online_backup_encoded[0] = 1
    elif OnlineBackup == "No internet service":
        online_backup_encoded[1] = 1
    else:
        online_backup_encoded[2] = 1

    device_protection_encoded = [0, 0, 0]
    if DeviceProtection == "No":
        device_protection_encoded[0] = 1
    elif DeviceProtection == "No internet service":
        device_protection_encoded[1] = 1
    else:
        device_protection_encoded[2] = 1

    tech_support_encoded = [0, 0, 0]
    if TechSupport == "No":
        tech_support_encoded[0] = 1
    elif TechSupport == "No internet service":
        tech_support_encoded[1] = 1
    else:
        tech_support_encoded[2] = 1

    streaming_tv_encoded = [0, 0, 0]
    if StreamingTV == "No":
        streaming_tv_encoded[0] = 1
    elif StreamingTV == "No internet service":
        streaming_tv_encoded[1] = 1
    else:
        streaming_tv_encoded[2] = 1

    streaming_movies_encoded = [0, 0, 0]
    if StreamingMovies == "No":
        streaming_movies_encoded[0] = 1
    elif StreamingMovies == "No internet service":
        streaming_movies_encoded[1] = 1
    else:
        streaming_movies_encoded[2] = 1

    contract_encoded = [0, 0, 0]
    if Contract == "Month-to-month":
        contract_encoded[0] = 1
    elif Contract == "One year":
        contract_encoded[1] = 1
    else:
        contract_encoded[2] = 1

    paperless_billing_encoded = [1, 0] if PaperlessBilling == "No" else [0, 1]

    payment_method_encoded = [0, 0, 0, 0]
    if PaymentMethod == "Bank transfer (automatic)":
        payment_method_encoded[0] = 1
    elif PaymentMethod == "Credit card (automatic)":
        payment_method_encoded[1] = 1
    elif PaymentMethod == "Electronic check":
        payment_method_encoded[2] = 1
    else:
        payment_method_encoded[3] = 1

    # Encode tenure group
    tenure_encoded = [0, 0, 0, 0, 0, 0]
    if tenure <= 12:
        tenure_encoded[0] = 1
    elif tenure <= 24:
        tenure_encoded[1] = 1
    elif tenure <= 36:
        tenure_encoded[2] = 1
    elif tenure <= 48:
        tenure_encoded[3] = 1
    elif tenure <= 60:
        tenure_encoded[4] = 1
    else:
        tenure_encoded[5] = 1

    # Combine all the encoded features into a single array
    input_data = [SeniorCitizen, MonthlyCharges, TotalCharges] + gender_encoded + partner_encoded + dependents_encoded + phone_service_encoded + multiple_lines_encoded + internet_service_encoded + online_security_encoded + online_backup_encoded + device_protection_encoded + tech_support_encoded + streaming_tv_encoded + streaming_movies_encoded + contract_encoded + paperless_billing_encoded + payment_method_encoded + tenure_encoded

    return np.array([input_data], dtype=float)

# Define the app
st.title("Customer Churn Prediction App")

st.header("Enter Customer Details:")

# Input fields for user data
SeniorCitizen = st.selectbox("Is the customer a senior citizen?", ["No", "Yes"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)
gender = st.selectbox("Gender", ["Female", "Male"])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
tenure = st.number_input("Tenure (months)", min_value=1, max_value=72)

# Convert categorical data to numerical format for prediction
input_data = preprocess_input(SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService, 
                              MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, 
                              StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure)

# Predict churn
if st.button("Predict Churn"):
    prediction = loaded_model.predict(input_data)
    if prediction[0] == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is not likely to churn.")