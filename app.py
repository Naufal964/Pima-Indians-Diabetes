# app.py
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SVM Diabetes Predictor", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model/svm_model.joblib")

model = load_model()

st.title("🩺 Prediksi Diabetes (SVM)")
st.write("Masukkan data pasien, lalu sistem memprediksi **Diabetes / Tidak** menggunakan **Support Vector Machine**.")

with st.sidebar:
    st.header("Tentang")
    st.markdown("""
- **Algoritma:** SVM (SVC)  
- **Dataset:** Pima Indians Diabetes (Kaggle)  
- **Deployment:** Streamlit  
    """)

st.subheader("Input Data Pasien")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=28.0)
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

if st.button("🔍 Prediksi"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # probabilitas kelas 1

    if pred == 1:
        st.error(f"✅ Hasil: **DIABETES** (probabilitas: {proba:.2f})")
    else:
        st.success(f"✅ Hasil: **TIDAK DIABETES** (probabilitas: {proba:.2f})")

    st.caption("Catatan: Ini hanya model ML berbasis data Kaggle, bukan diagnosis medis.")
