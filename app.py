from pathlib import Path
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="SVM Diabetes Predictor", page_icon="🩺")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "svm_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🩺 Prediksi Diabetes (SVM)")
st.write("Masukkan data pasien, lalu klik Prediksi")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("BloodPressure", 0, 200, 70)
    skin_thickness = st.number_input("SkinThickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 80.0, 28.0)
    dpf = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

input_data = np.array([[pregnancies, glucose, blood_pressure,
                        skin_thickness, insulin, bmi, dpf, age]])

if st.button("🔍 Prediksi"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"DIABETES (probabilitas {proba:.2f})")
    else:
        st.success(f"TIDAK DIABETES (probabilitas {proba:.2f})")
