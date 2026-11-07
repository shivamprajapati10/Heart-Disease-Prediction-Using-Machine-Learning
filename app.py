import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load the Random Forest model.")
    st.error(str(e))
    st.stop()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("❤️ Heart Disease Prediction System")

st.markdown(
    """
    ### Machine Learning Based Prediction

    Enter the patient's medical information below and click
    **Predict Heart Disease** to get the model's prediction.
    """
)

st.divider()


# --------------------------------------------------
# GET FEATURE NAMES
# --------------------------------------------------

if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    st.error(
        "The saved model does not contain feature names. "
        "Please provide the exact feature columns used during training."
    )
    st.stop()


st.subheader("🩺 Patient Information")

st.write(f"Model expects **{len(feature_names)} features**.")


# --------------------------------------------------
# CREATE INPUT FIELDS
# --------------------------------------------------

# --------------------------------------------------
# CREATE INPUT FIELDS
# --------------------------------------------------

inputs = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):

    display_name = feature.replace("_", " ").title()

    if i % 2 == 0:
        current_col = col1
    else:
        current_col = col2

    with current_col:

        # Special handling for SEX
        if feature.lower() == "sex":

            sex = st.selectbox(
                "Sex",
                ["Female", "Male"]
            )

            # Convert to the format used during model training
            if sex == "Female":
                inputs[feature] = 0
            else:
                inputs[feature] = 1

        else:

            inputs[feature] = st.number_input(
                display_name,
                value=0 
            )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔍 Predict Heart Disease",
    type="primary",
    use_container_width=True
):

    # Create DataFrame in EXACT model feature order
    input_data = pd.DataFrame(
        [inputs],
        columns=feature_names
    )

    st.subheader("📊 Prediction Result")

    try:

        prediction = model.predict(input_data)[0]

        # Probability if supported
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0]

            # Probability of class 1
            if len(probability) == 2:
                disease_probability = probability[1] * 100
            else:
                disease_probability = max(probability) * 100
        else:
            disease_probability = None


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Prediction: Heart Disease Detected"
            )

            if disease_probability is not None:
                st.metric(
                    "Estimated Probability",
                    f"{disease_probability:.2f}%"
                )

        else:

            st.success(
                "✅ Prediction: No Heart Disease Detected"
            )

            if disease_probability is not None:
                st.metric(
                    "Estimated Probability of Heart Disease",
                    f"{disease_probability:.2f}%"
                )


        # --------------------------------------------------
        # SHOW INPUT DATA
        # --------------------------------------------------

        with st.expander("View Patient Input Data"):
            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.code(str(e))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Heart Disease Prediction • Random Forest Machine Learning Model"
)

st.caption(
    "⚠️ This application is for educational purposes and is not a medical diagnosis."
)