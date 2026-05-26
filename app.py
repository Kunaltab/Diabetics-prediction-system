from pathlib import Path
import pickle

import numpy as np
import streamlit as st

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

MODEL_PATH = Path(__file__).resolve().parent / "diabetes_model.pkl"


@st.cache_resource
def load_model(model_path: Path):
    try:
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    except Exception as exc:
        st.error(
            "Could not load `diabetes_model.pkl`. Run `python diabetes_model.py` to retrain "
            "the model in your current environment."
        )
        st.exception(exc)
        st.stop()


def predict_diabetes(features):
    model = load_model(MODEL_PATH)
    data = np.array(features, dtype=float).reshape(1, -1)
    prediction = int(model.predict(data)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(data)[0][1])

    return prediction, probability


if not MODEL_PATH.exists():
    st.error("Model file `diabetes_model.pkl` not found. Run `python diabetes_model.py` first.")
    st.stop()

st.title("Diabetes Prediction App")
st.caption("Enter patient values below to estimate diabetes risk from the trained model.")

left_col, right_col = st.columns(2)

with left_col:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with right_col:
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=32.0, format="%.1f")
    dpf = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, format="%.2f"
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

if st.button("Predict"):
    features = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
    result, probability = predict_diabetes(features)

    if result == 1:
        st.error("Prediction: Likely diabetic")
    else:
        st.success("Prediction: Not likely diabetic")

    if probability is not None:
        st.info(f"Estimated probability of diabetes: {probability:.1%}")

st.warning("This app is for educational use only and is not a medical diagnosis tool.")
