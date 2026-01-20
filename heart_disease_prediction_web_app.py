# -*- coding: utf-8 -*-
import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('D:\ML\trained_model_heart_disease.sav', 'rb'))

# Prediction function
def heart_disease_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'YOU DO NOT HAVE HEART DISEASE'
    else:
        return 'YOU HAVE HEART DISEASE'

def main():

    st.title('Heart Disease Prediction Web App')

    age = st.text_input('Your Age')
    sex = st.text_input('Your Sex')
    cp = st.text_input('Chest Pain Type')
    trestbps = st.text_input('Resting Blood Pressure Value')
    chol = st.text_input('Cholesterol Value')
    fbs = st.text_input('Fasting Blood Sugar Value')
    restecg = st.text_input('Resting Electrocardiographic Results Value')
    thalach = st.text_input('Maximum Heart Rate Achieved')
    exang = st.text_input('Exercise Induced Angina')
    oldpeak = st.text_input('ST Depression')
    slope = st.text_input('Slope of the Peak Exercise ST Segment')
    ca = st.text_input('Number of Major Vessels')
    thal = st.text_input('Thalassemia')

    diagnosis = ''

    if st.button('Heart Disease Test Result'):
        diagnosis = heart_disease_prediction([
         age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
        ])

    st.success(diagnosis)

if __name__ == '__main__':
    main()

