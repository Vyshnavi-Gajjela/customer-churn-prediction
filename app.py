
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/xgb_model.pkl')
training_columns = joblib.load('models/training_columns.pkl')

st.title('Customer Churn Prediction System')
st.subheader('Predict Customer Churn Probability')

tenure = st.slider('Tenure',0,72,12)

monthlycharges = st.slider('Monthly Charges', 0,150,70)

totalcharges = st.slider('Total Charges',0,10000,2500)

seniorcitizen = st.selectbox('Senior Citizen',[0,1])

input_data = pd.DataFrame({
    'SeniorCitizen':[seniorcitizen],
    'tenure':[tenure],
    'MonthlyCharges':[monthlycharges],
    'TotalCharges':[totalcharges],
    'AvgMonthlySpend':[
        totalcharges/(tenure+1)]
})

for column in training_columns:
    if column not in input_data.columns:
        input_data[column] = 0

input_data = input_data.reindex(columns=training_columns,fill_value=0)

if st.button('Predict Churn'):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    st.subheader(f'Churn Probability: {probability:.2f}')

    if probability < 0.3:
        st.success('Low Risk Customer')

    elif probability < 0.7:
        st.warning('Medium Risk Customer')

    else:
        st.error('High Risk Customer')

    if monthlycharges > 80:
        st.info('Suggested Action: Provide Loyalty Discount')

    elif tenure < 12:
        st.info('Suggested Action: Offer Onboarding Support')

    else:
        st.info('Suggested Action: Recommend Annual Subscription Plan')
