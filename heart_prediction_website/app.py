import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("ann_model.joblib")

model = load_model()

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the dataset measurements below and select **Predict**.")
st.info("Educational project only. This model output is not a medical diagnosis and must not be used for treatment decisions.")

# Choices are taken from the training dataset.
choices = {'sex': ['female', 'male'], 'cp': ['asympt', 'atyp_angina', 'non_anginal', 'typ_angina'], 'fbs': ['f', 't'], 'restecg': ['left_vent_hyper', 'normal', 'st_t_wave_abnormality'], 'exang': ['no', 'yes'], 'slope': ['down', 'flat', 'up'], 'thal': ['fixed_defect', 'normal', 'reversable_defect']}
ranges = {'age': {'min': 29.0, 'max': 77.0, 'median': 55.0}, 'trestbps': {'min': 94.0, 'max': 200.0, 'median': 130.0}, 'chol': {'min': 126.0, 'max': 564.0, 'median': 240.0}, 'thalach': {'min': 71.0, 'max': 202.0, 'median': 153.0}, 'oldpeak': {'min': 0.0, 'max': 6.2, 'median': 0.8}, 'ca': {'min': 0.0, 'max': 3.0, 'median': 0.0}}

st.subheader("Patient measurements")
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    values = {}

    # Numeric and categorical fields in the same order as the training data.
    for i, (col, info) in enumerate(ranges.items()):
        target = c1 if i % 2 == 0 else c2
        with target:
            label = {
                "age":"Age (years)",
                "trestbps":"Resting blood pressure",
                "chol":"Serum cholesterol",
                "thalach":"Maximum heart rate",
                "oldpeak":"ST depression (oldpeak)"
            }.get(col, col)
            step = 0.1 if col == "oldpeak" else 1.0
            values[col] = st.number_input(
                label, min_value=float(info["min"]), max_value=float(info["max"]),
                value=float(info["median"]), step=step
            )

    st.markdown("---")
    st.caption("Select the categories used by the original dataset.")
    c1, c2 = st.columns(2)
    for i, (col, opts) in enumerate(choices.items()):
        target = c1 if i % 2 == 0 else c2
        label = {
            "sex":"Sex",
            "cp":"Chest pain type",
            "fbs":"Fasting blood sugar",
            "restecg":"Resting ECG",
            "exang":"Exercise-induced angina",
            "slope":"ST slope",
            "thal":"Thal",
            "ca":"Number of major vessels (ca)"
        }.get(col, col)
        with target:
            values[col] = st.selectbox(label, opts, key=col)

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    # Match the exact feature order used during model training.
    input_df = pd.DataFrame([values])
    prediction = int(model.predict(input_df)[0])
    st.subheader("Model output")
    if prediction == 0:
        st.success("Predicted class: 0")
        st.write("In the original project mapping, class 0 corresponds to the dataset label <50.")
    else:
        st.warning("Predicted class: 1")
        st.write("In the original project mapping, class 1 corresponds to the dataset label >50_1.")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        st.write("Model probability estimates (not clinical risk):")
        st.write(f"Class 0: {probs[0]*100:.1f}%")
        st.write(f"Class 1: {probs[1]*100:.1f}%")

    st.caption("The class labels follow the target encoding in the supplied notebook: <50 = 0 and >50_1 = 1.")
