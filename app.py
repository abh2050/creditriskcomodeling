import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import sklearn

# Load the pre-trained GBT model
model_filename = 'finalized_model.sav'
model = pickle.load(open(model_filename, 'rb'))

# Title and description
st.title("Loan Default Prediction with Gradient Boosted Trees")
st.write("This app predicts the probability of loan default based on loan characteristics.")

# Define the expected features based on the model's structure
expected_features = [
    'person_income', 'loan_int_rate', 'loan_percent_income',
    'loan_amnt', 'person_home_ownership_MORTGAGE', 'loan_grade_F'
]

# Section to gather user inputs
st.header("Input Loan Details")

# User inputs for required features
loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=50000, step=1000)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=1.0, max_value=25.0, step=0.1)
person_income = st.number_input("Person's Annual Income ($)", min_value=10000, max_value=100000, step=1000)

# Derived features and fixed values
loan_percent_income = loan_amnt / person_income
person_home_ownership_MORTGAGE = st.checkbox("Mortgage on Home?", value=False)
loan_grade_F = st.checkbox("Is Loan Grade F?", value=False)

# Create a dictionary for the input data
input_data = {
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'person_income': person_income,
    'loan_percent_income': loan_percent_income,
    'person_home_ownership_MORTGAGE': person_home_ownership_MORTGAGE,
    'loan_grade_F': loan_grade_F,
}

# Convert the dictionary to a DataFrame and reorder the columns to match the expected features
input_df = pd.DataFrame([input_data])[expected_features]

# Use the model to predict the probability of default
prediction = model.predict_proba(input_df)[0][1]  # Probability of default

# Display the prediction
st.header("Loan Default Prediction")
st.write(f"The predicted probability of loan default is: {prediction:.2%}")

# Additional information
if prediction > 0.4:
    st.warning("This loan is likely to default.")
else:
    st.success("This loan is less likely to default.")
