import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/xgb_model.pkl')
training_columns = joblib.load('models/training_columns.pkl')

st.set_page_config(
    page_title='Customer Churn Prediction',
    layout='centered'
)

st.title('Customer Churn Prediction System')
st.subheader('Predict Customer Churn Probability')
st.markdown('This application predicts telecom customer churn probability using an XGBoost Machine Learning model.')
tenure = st.slider('Tenure (Months)', 0, 72, 12)

monthlycharges = st.slider('Monthly Charges', 0, 150, 70)

totalcharges = st.slider('Total Charges', 0, 10000, 2500)

seniorcitizen = st.selectbox('Senior Citizen', [0, 1])

input_data = pd.DataFrame({
    'SeniorCitizen': [seniorcitizen],
    'tenure': [tenure],
    'MonthlyCharges': [monthlycharges],
    'TotalCharges': [totalcharges],
    'AvgMonthlySpend': [totalcharges / (tenure + 1)]
})

for column in training_columns:
    if column not in input_data.columns:
        input_data[column] = 0

input_data = input_data.reindex(
    columns=training_columns,
    fill_value=0
)

if st.button('Predict Churn'):
    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    st.subheader(f'Churn Probability: {probability:.2f}')

    if probability < 0.30:
        st.success('Low Risk Customer')
        st.info('Suggested Action: Maintain engagement through loyalty rewards and personalized offers.')

    elif probability < 0.70:
        st.warning('Medium Risk Customer')
        if tenure < 12:
            st.info('Suggested Action: Provide onboarding assistance and proactive customer support.')

        elif monthlycharges > 80:
            st.info('Suggested Action: Offer customized discounts or bundled service plans.')

        else:
            st.info('Suggested Action: Improve engagement through targeted retention campaigns.')

    else:
        st.error('High Risk Customer')

        if monthlycharges > 80:
            st.info('Suggested Action: Immediate retention intervention with premium discount offers.')

        elif tenure < 12:
            st.info('Suggested Action: Assign dedicated onboarding and customer success support.')

        else:
            st.info('Suggested Action: Recommend long-term contract plans and personalized retention incentives.')