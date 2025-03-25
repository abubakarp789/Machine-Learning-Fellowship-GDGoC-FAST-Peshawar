import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    color: #333;
}
.highlight {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("Heart Disease Severity Predictor")
st.markdown("### Predict the Likelihood of Heart Disease 🩺", unsafe_allow_html=True)
st.write("Use this tool to assess potential heart disease risk based on medical parameters.")

# Load the saved model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Input Feature Collection
def collect_input_features():
    col1, col2 = st.columns(2)
    
    with col1:
        dataset = st.selectbox("Dataset Origin", ["Cleveland", "Other"])
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", [
            "Typical Angina", 
            "Atypical Angina", 
            "Non-anginal Pain", 
            "Asymptomatic"
        ])
        resting_bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=220)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    
    with col2:
        max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        st_depression = st.number_input("ST Depression", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
        target_num = st.number_input("Target Num (Heart Disease Severity)", min_value=0, max_value=2, value=0)
        rest_ecg = st.selectbox("Resting ECG", [
            "Normal", 
            "ST-T Wave Abnormality", 
            "Left Ventricular Hypertrophy"
        ])
        slope = st.selectbox("ST Slope", [
            "Upsloping", 
            "Flat", 
            "Downsloping"
        ])
        thal = st.selectbox("Thalassemia", [
            "Normal", 
            "Fixed Defect", 
            "Reversible Defect"
        ])
    
    return {
        'id': age,  # Using age as a placeholder for id
        'dataset': 1 if dataset == "Cleveland" else 0,
        'age': age,
        'trestbps': resting_bp,
        'chol': cholesterol,
        'fbs': 1 if fbs == "Yes" else 0,
        'thalch': max_heart_rate,
        'exang': 1 if exercise_induced_angina == "Yes" else 0,
        'oldpeak': st_depression,
        'ca': ca,
        'num': target_num,
        'sex_Female': 1 if sex == "Female" else 0,
        'sex_Male': 1 if sex == "Male" else 0,
        'cp_asymptomatic': 1 if chest_pain == "Asymptomatic" else 0,
        'cp_atypical angina': 1 if chest_pain == "Atypical Angina" else 0,
        'cp_non-anginal': 1 if chest_pain == "Non-anginal Pain" else 0,
        'cp_typical angina': 1 if chest_pain == "Typical Angina" else 0,
        'restecg_lv hypertrophy': 1 if rest_ecg == "Left Ventricular Hypertrophy" else 0,
        'restecg_normal': 1 if rest_ecg == "Normal" else 0,
        'restecg_st-t abnormality': 1 if rest_ecg == "ST-T Wave Abnormality" else 0,
        'slope_downsloping': 1 if slope == "Downsloping" else 0,
        'slope_flat': 1 if slope == "Flat" else 0,
        'slope_upsloping': 1 if slope == "Upsloping" else 0,
        'thal_fixed defect': 1 if thal == "Fixed Defect" else 0,
        'thal_normal': 1 if thal == "Normal" else 0,
        'thal_reversable defect': 1 if thal == "Reversible Defect" else 0
    }

# Preprocessing Function
def preprocess_input(input_data):
    # Create DataFrame with all 25 features, ensuring the exact same order
    feature_order = [
        'id', 'dataset', 'age', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 
        'oldpeak', 'ca', 'num', 'sex_Female', 'sex_Male', 
        'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
        'restecg_lv hypertrophy', 'restecg_normal', 'restecg_st-t abnormality',
        'slope_downsloping', 'slope_flat', 'slope_upsloping',
        'thal_fixed defect', 'thal_normal', 'thal_reversable defect'
    ]
    
    # Create DataFrame with zeros for all features
    df = pd.DataFrame(0.0, index=[0], columns=feature_order)
    
    # Update with input data
    for feature, value in input_data.items():
        if feature in df.columns:
            df.loc[0, feature] = value
    
    # Select features for the model (excluding target)
    X = df.drop('num', axis=1)
    
    return X

# Prediction Function
def predict_heart_disease(input_features):
    preprocessed_features = preprocess_input(input_features)
    
    # Debug print to check features
    st.write("Preprocessed Features:")
    st.dataframe(preprocessed_features)
    st.write("Number of Features:", len(preprocessed_features.columns))
    st.write("Feature Names:", list(preprocessed_features.columns))
    
    prediction = model.predict(preprocessed_features)
    return prediction[0]

# Prediction Interpretation
def interpret_prediction(prediction):
    severity_map = {
        0: "Low Risk (No Significant Heart Disease)",
        1: "Moderate Risk (Potential Heart Disease)",
        2: "High Risk (Significant Heart Disease)"
    }
    return severity_map.get(prediction, "Unknown Risk")

# Main Prediction Interface
if model:
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    
    # Feature Input
    st.subheader("Enter Patient Details")
    input_features = collect_input_features()
    
    # Prediction Button
    if st.button("Predict Heart Disease Risk", type="primary"):
        with st.spinner('Analyzing Data...'):
            prediction = predict_heart_disease(input_features)
            risk_interpretation = interpret_prediction(prediction)
        
        # Result Display
        st.markdown(f"### Prediction Result: {risk_interpretation}", unsafe_allow_html=True)
        
        # Risk Advice
        if prediction == 0:
            st.success("🟢 Good news! Low risk of significant heart disease. Continue maintaining a healthy lifestyle.")
        elif prediction == 1:
            st.warning("🟠 Moderate risk detected. Consult with a healthcare professional for further evaluation.")
        else:
            st.error("🔴 High risk identified. Immediate medical consultation is recommended.")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Model could not be loaded. Please check the model file.")

# Footer
st.markdown("---")
st.markdown("*Disclaimer: This is a predictive tool and should not replace professional medical advice.*")