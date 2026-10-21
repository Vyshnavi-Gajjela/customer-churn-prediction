"""Streamlit app for the saved end-to-end churn pipeline."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
st.set_page_config(page_title="Customer Churn Risk", page_icon="📊", layout="centered")
st.title("Customer Churn Prediction & Retention Support")
st.caption("A model-based risk estimate for prioritizing review—not a guarantee that a customer will leave.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train.py` from the project folder first.")
    st.stop()
model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
st.caption(f"Model selected in latest training run: {metadata.get('best_model', 'unknown')}")

with st.form("customer_form"):
    st.subheader("Customer profile")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Has partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No phone service", "Yes", "No"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    with col2:
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=2000.0, value=70.0, step=1.0)
        total = st.number_input("Total charges ($)", min_value=0.0, max_value=100000.0, value=float(monthly * tenure), step=10.0)
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    submitted = st.form_submit_button("Estimate churn risk", type="primary")

if submitted:
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": int(senior), "Partner": partner, "Dependents": dependents,
        "tenure": int(tenure), "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": float(monthly), "TotalCharges": float(total),
    }])
    probability = float(model.predict_proba(row)[0, 1])
    threshold = float(metadata.get("thresholds_tuned_on_validation", {}).get(metadata.get("best_model", ""), {}).get("threshold_selected_on_validation", 0.5))
    risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.30 else "Low"
    st.subheader("Prediction")
    st.metric("Estimated churn probability", f"{probability:.1%}")
    st.write(f"**Risk band:** {risk}")
    st.progress(min(max(probability, 0.0), 1.0))
    if probability >= threshold:
        st.warning("The model classifies this customer as likely to churn at the validation-selected operating threshold. Consider reviewing the account for an appropriate retention intervention.")
    else:
        st.info("The model does not classify this customer as likely to churn at the validation-selected operating threshold. Continue normal customer monitoring.")
    st.caption("The risk band uses probability bands (<30%, 30–<70%, ≥70%). The model's classification threshold is selected separately on validation data.")
    with st.expander("Input data used"):
        st.dataframe(row, use_container_width=True)
