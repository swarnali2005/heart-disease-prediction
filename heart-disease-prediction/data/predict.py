import joblib
import numpy as np

def predict_patient(patient_data: dict, model_name='random_forest'):
    scaler = joblib.load('../models/scaler.pkl')
    model  = joblib.load(f'../models/{model_name}.pkl')

    features = [
        patient_data['age'], patient_data['sex'], patient_data['cp'],
        patient_data['trestbps'], patient_data['chol'], patient_data['fbs'],
        patient_data['restecg'], patient_data['thalach'], patient_data['exang'],
        patient_data['oldpeak'], patient_data['slope'], patient_data['ca'],
        patient_data['thal']
    ]

    X = scaler.transform([features])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)

    print("Raw probabilities:", probability)  # Debug line

    confidence = probability[0][1]

    return {
        'prediction': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
        'confidence': f'{confidence * 100:.1f}%'
    }

# Example usage
if __name__ == '__main__':
    sample = {
        'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
        'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
    }
    result = predict_patient(sample)
    print(result)