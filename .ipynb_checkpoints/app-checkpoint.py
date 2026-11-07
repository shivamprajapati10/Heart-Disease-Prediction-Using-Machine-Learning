import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("random_forest_model.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details below.")

# User Inputs
age = st.number_input("Age", 18, 100, 40)
sex = st.selectbox("Sex", [0, 1])

chest_pain_type = st.selectbox("Chest Pain Type", [0,1,2,3])

resting_bp = st.number_input("Resting Blood Pressure", 80,250,120)

cholesterol = st.number_input("Cholesterol",100,500,200)

fasting_blood_sugar = st.selectbox(
    "Fasting Blood Sugar",
    [0,1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    [0,1,2]
)

max_heart_rate = st.number_input(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    [0,1]
)

oldpeak = st.number_input(
    "Oldpeak",
    0.0,
    10.0,
    1.0
)

st_slope = st.selectbox(
    "ST Slope",
    [0,1,2]
)

num_major_vessels = st.selectbox(
    "Number of Major Vessels",
    [0,1,2,3]
)

thalassemia = st.selectbox(
    "Thalassemia",
    [1,2,3]
)

bmi = st.number_input(
    "BMI",
    10.0,
    50.0,
    25.0
)

smoking = st.selectbox(
    "Smoking",
    [0,1]
)

diabetes = st.selectbox(
    "Diabetes",
    [0,1]
)

hypertension = st.selectbox(
    "Hypertension",
    [0,1]
)

family_history = st.selectbox(
    "Family History",
    [0,1]
)

physical_activity = st.number_input(
    "Physical Activity (hours/week)",
    0.0,
    20.0,
    3.0
)

# Prediction Button
if st.button("Predict"):

    input_data = pd.DataFrame([[
        age,
        sex,
        chest_pain_type,
        resting_bp,
        cholesterol,
        fasting_blood_sugar,
        resting_ecg,
        max_heart_rate,
        exercise_angina,
        oldpeak,
        st_slope,
        num_major_vessels,
        thalassemia,
        bmi,
        smoking,
        diabetes,
        hypertension,
        family_history,
        physical_activity
    ]], columns=[
        "age",
        "sex",
        "chest_pain_type",
        "resting_bp",
        "cholesterol",
        "fasting_blood_sugar",
        "resting_ecg",
        "max_heart_rate",
        "exercise_angina",
        "oldpeak",
        "st_slope",
        "num_major_vessels",
        "thalassemia",
        "bmi",
        "smoking",
        "diabetes",
        "hypertension",
        "family_history",
        "physical_activity"
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")