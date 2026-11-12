from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


FEATURES = [
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
    "physical_activity",
]


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().with_name("random_forest_model.pkl")
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as error:
    st.error("Could not load the Random Forest model.")
    st.exception(error)
    st.stop()

feature_names = getattr(model, "feature_names_in_", None)
if feature_names is None:
    st.error("The saved model does not expose its training feature names.")
    st.stop()

feature_names = list(feature_names)
if len(feature_names) != len(FEATURES) or set(feature_names) != set(FEATURES):
    st.error(
        "The saved model's feature schema does not match the inputs configured "
        "in this app."
    )
    st.stop()


st.title("❤️ Heart Disease Prediction")
st.markdown(
    "Enter the patient information below to get an educational prediction "
    "from the trained model."
)
# st.info(
#     "Numeric input ranges match the values represented in this project's "
#     "training data. The source data does not document all categorical code "
#     "meanings, so category labels use common mappings and should be confirmed "
#     "against the training pipeline."
# )

inputs = {}


def select_code(feature, label, options, help_text):
    selected = st.selectbox(
        label,
        list(options),
        index=None,
        placeholder="Select an option",
        key=feature,
        help=help_text,
    )
    inputs[feature] = options[selected] if selected is not None else None


def yes_no_code(feature, label, help_text):
    select_code(
        feature,
        label,
        {"No": 0, "Yes": 1},
        help_text,
    )


st.subheader("Demographics")
age_col, sex_col, bmi_col = st.columns(3)
with age_col:
    inputs["age"] = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=90,
        value=None,
        step=1,
        help="Age range represented in the training data: 18–90 years.",
    )
with sex_col:
    select_code(
        "sex",
        "Sex",
        {"Female": 0, "Male": 1},
        "Encoded as Female = 0 and Male = 1.",
    )
with bmi_col:
    inputs["bmi"] = st.number_input(
        "BMI (kg/m²)",
        min_value=16.0,
        max_value=40.0,
        value=None,
        step=0.1,
        help="BMI range represented in the training data: 16–40.",
    )

st.subheader("Symptoms and exercise")
symptom_col, exercise_col = st.columns(2)
with symptom_col:
    select_code(
        "chest_pain_type",
        "Chest pain type",
        {
            "Typical angina": 0,
            "Atypical angina": 1,
            "Non-anginal pain": 2,
            "Asymptomatic": 3,
        },
        "Select the closest description of the patient's chest pain.",
    )
    yes_no_code(
        "exercise_angina",
        "Chest pain during exercise",
        "Whether angina is triggered by physical exertion.",
    )
    inputs["max_heart_rate"] = st.number_input(
        "Maximum heart rate (bpm)",
        min_value=70,
        max_value=202,
        value=None,
        step=1,
        help="Maximum heart-rate range represented in the training data: 70–202 bpm.",
    )
with exercise_col:
    inputs["oldpeak"] = st.number_input(
        "Oldpeak (ST depression)",
        min_value=0.0,
        max_value=6.0,
        value=None,
        step=0.1,
        help="ST depression induced by exercise relative to rest; training range: 0–6.",
    )
    select_code(
        "st_slope",
        "ST segment slope during peak exercise",
        {"Upsloping": 0, "Flat": 1, "Downsloping": 2},
        "Select the observed slope.",
    )

st.subheader("Clinical measurements")
measurement_col, test_col = st.columns(2)
with measurement_col:
    inputs["resting_bp"] = st.number_input(
        "Resting blood pressure (mmHg)",
        min_value=90,
        max_value=180,
        value=None,
        step=1,
        help="Resting blood-pressure range represented in the training data: 90–180 mmHg.",
    )
    inputs["cholesterol"] = st.number_input(
        "Serum cholesterol (mg/dL)",
        min_value=120,
        max_value=360,
        value=None,
        step=1,
        help="Cholesterol range represented in the training data: 120–360 mg/dL.",
    )
    yes_no_code(
        "fasting_blood_sugar",
        "Fasting blood sugar above 120 mg/dL",
        "Select Yes when fasting blood sugar is above 120 mg/dL.",
    )
with test_col:
    select_code(
        "resting_ecg",
        "Resting ECG",
        {
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Left ventricular hypertrophy": 2,
        },
        "Choose the closest resting ECG result.",
    )
    inputs["num_major_vessels"] = st.selectbox(
        "Major vessels colored by fluoroscopy",
        [0, 1, 2, 3],
        index=None,
        placeholder="Select number of vessels",
        help="Number of major vessels (0–3) colored by fluoroscopy.",
    )
    select_code(
        "thalassemia",
        "Thalassemia test result",
        {
            "Normal": 1,
            "Fixed defect": 2,
            "Reversible defect": 3,
        },
        "Choose the recorded thalassemia test result.",
    )

st.subheader("Medical history and lifestyle")
history_col, lifestyle_col = st.columns(2)
with history_col:
    yes_no_code("smoking", "Current smoker", "Current smoking status.")
    yes_no_code("hypertension", "Diagnosed hypertension", "History of hypertension.")
    yes_no_code("diabetes", "Diagnosed diabetes", "History of diabetes.")
    yes_no_code(
        "family_history",
        "Family history of cardiovascular disease",
        "Whether the patient has a family history of cardiovascular disease.",
    )
with lifestyle_col:
    inputs["physical_activity"] = st.number_input(
        "Physical activity level",
        min_value=0.0,
        max_value=10.0,
        value=None,
        step=0.1,
        help="Use 0 for sedentary and 10 for very active; the dataset uses a 0–10 scale.",
    )

st.divider()
if st.button(
    "🔍 Predict Heart Disease",
    type="primary",
    width="stretch",
):
    missing_features = [
        feature for feature in feature_names if inputs.get(feature) is None
    ]
    if missing_features:
        st.warning("Please fill in all fields before requesting a prediction.")
        st.stop()

    input_data = pd.DataFrame(
        [[inputs[feature] for feature in feature_names]],
        columns=feature_names,
    )

    try:
        prediction = model.predict(input_data)[0]
        positive_probability = None
        if hasattr(model, "predict_proba"):
            class_index = list(model.classes_).index(1)
            positive_probability = model.predict_proba(input_data)[0][class_index] * 100
    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)
        st.stop()

    st.subheader("Prediction result")
    if prediction == 1:
        st.error("Model prediction: Heart disease detected")
    else:
        st.success("Model prediction: No heart disease detected")

    if positive_probability is not None:
        st.metric(
            "Model-estimated probability for heart disease",
            f"{positive_probability:.2f}%",
        )

    with st.expander("View patient input data"):
        st.dataframe(input_data, width="stretch")

st.divider()
st.caption(
    "For educational use only. This model is not a medical diagnosis or "
    "substitute for advice from a qualified healthcare professional."
)
