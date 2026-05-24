import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Customer Churn Prediction',layout='wide')

st.title('Customer Churn Prediction System')

st.write('Predict whether a customer is likely to churn or not.')

# LOAD SCALER
scaler = joblib.load('models/scaler.pkl')

# LOAD FEATURE NAMES
feature_names = joblib.load('models/feature_names.pkl')

# LOAD BEST MODEL
model = joblib.load('models/best_model.pkl')

# USER INPUTS
SeniorCitizen = st.selectbox('Senior Citizen',[0, 1])

tenure = st.slider('Tenure (Months)',0,72,12)

InternetService = st.selectbox('Internet Service',['DSL', 'Fiber optic', 'No'])

Contract = st.selectbox('Contract Type',['Month-to-month', 'One year', 'Two year'])

PaymentMethod = st.selectbox(
    'Payment Method',
    [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ]
)

MonthlyCharges = st.number_input('Monthly Charges',min_value=0.0,value=70.0)

TotalCharges = st.number_input('Total Charges',min_value=0.0,value=1000.0)

TotalServices = st.slider('Total Services Used',0,6,3)

# PREDICT BUTTON
if st.button('Predict Churn'):

    AvgMonthlySpend = TotalCharges / (tenure + 1)

    HighValueCustomer = int(MonthlyCharges > 70)

    MonthlyContractRisk = int(Contract == 'Month-to-month')

    input_data = pd.DataFrame({
        'SeniorCitizen': [SeniorCitizen],
        'tenure': [tenure],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'AvgMonthlySpend': [AvgMonthlySpend],
        'TotalServices': [TotalServices],
        'HighValueCustomer': [HighValueCustomer],
        'MonthlyContractRisk': [MonthlyContractRisk],

        'InternetService_Fiber optic': [1 if InternetService == 'Fiber optic' else 0],

        'InternetService_No': [1 if InternetService == 'No' else 0],

        'Contract_One year': [1 if Contract == 'One year' else 0],

        'Contract_Two year': [1 if Contract == 'Two year' else 0],

        'PaymentMethod_Credit card (automatic)': [1 if PaymentMethod == 'Credit card (automatic)' else 0],

        'PaymentMethod_Electronic check': [1 if PaymentMethod == 'Electronic check' else 0],

        'PaymentMethod_Mailed check': [1 if PaymentMethod == 'Mailed check' else 0]
    })

    input_data = input_data.reindex(columns=feature_names,fill_value=0)

    input_scaled = scaler.transform(input_data)

    churn_probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(churn_probability >= 0.5)

    st.subheader('Prediction Result')

    if prediction == 1:
        st.error(f'Customer is likely to churn with probability {churn_probability:.2%}')
        st.subheader('Recommendations')
        st.write('- Offer long-term contract discounts')
        st.write('- Improve customer support quality')
        st.write('- Provide personalized retention offers')
        st.write('- Reduce service-related complaints')

    else:
        st.success(f'Customer is likely to stay with probability {(1 - churn_probability):.2%}')
        st.subheader('Recommendations')
        st.write('- Maintain customer engagement')
        st.write('- Continue quality support services')
        st.write('- Provide loyalty rewards')