import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

# Streamlit app title
st.title('Credit Card Fraud Detection System')

# Sidebar for user inputs
st.sidebar.header('Transaction Input Features')

# Function to take user input from the sidebar
def user_input_features():
    time = st.sidebar.number_input('Time since first transaction', min_value=0)
    amount = st.sidebar.number_input('Transaction Amount', min_value=0.0, format="%.2f")
    features = {}
    for i in range(1, 29):
        features[f'V{i}'] = st.sidebar.number_input(f'V{i}', value=0.0, format="%.2f")
    
    input_data = pd.DataFrame([{
        'Time': time,
        'Amount': amount,
        **features
    }])
    return input_data

# Input from user
input_df = user_input_features()

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_df)  # Predicting using the loaded model
    prediction_proba = model.predict_proba(input_df)

    # Display the result
    if prediction[0] == 1:
        st.error(f'Fraudulent Transaction Detected! Probability: {prediction_proba[0][1]:.2f}')
    else:
        st.success(f'Transaction is Legitimate. Probability of Fraud: {prediction_proba[0][1]:.2f}')
