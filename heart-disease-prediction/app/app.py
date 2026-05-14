import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀")

st.title("🫀 Heart Disease Prediction")
st.markdown("Enter patient details below to predict the risk of heart disease.")

scaler = joblib.load('../models/scaler.pkl')

col1, col2, col3 = st.columns(3)

with col1:
    age      = st.slider("Age", 20, 80, 54)
    sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol     = st.number_input("Cholesterol", 100, 600, 240)

with col2:
    fbs      = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg  = st.selectbox("Resting ECG", [0, 1, 2])
    thalach  = st.number_input("Max Heart Rate", 60, 220, 150)
    exang    = st.selectbox("Exercise Induced Angina", [0, 1])

with col3:
    oldpeak    = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope      = st.selectbox("Slope", [0, 1, 2])
    ca         = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    thal       = st.selectbox("Thalassemia", [0, 1, 2, 3])
    model_name = st.selectbox("Select Model", ['random_forest', 'xgboost', 'svm', 'logistic_regression'])

if st.button("🔍 Predict"):
    model    = joblib.load(f'../models/{model_name}.pkl')
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    X        = scaler.transform([features])

    prediction  = model.predict(X)[0]
    confidence  = model.predict_proba(X)[0][1] * 100

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ Heart Disease Detected — Confidence: {confidence:.1f}%")
    else:
        st.success(f"✅ No Heart Disease — Confidence: {confidence:.1f}%")