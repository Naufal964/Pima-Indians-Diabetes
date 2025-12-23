# app.py
from pathlib import Path
import numpy as np
import streamlit as st
import joblib

st.set_page_config(page_title="SVM Diabetes Predictor", page_icon="🩺", layout="centered")

# Buat path absolut berdasarkan lokasi file app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "svm_model.joblib"

@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan di: {model_path}\n"
            "Pastikan kamu sudah menjalankan train.py dan folder 'model/' ikut ter-upload."
        )
    return joblib.load(model_path)

# Load model dengan error handling agar pesan jelas
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("Model gagal dimuat ❌")
    st.code(str(e))
    st.stop()

st.title("🩺 Prediksi Diabetes (SVM)")
st.write("Masukkan data pasien, lalu sistem memprediksi **Diabetes / Tidak** menggunakan **Support Vector Machine**.")

with st.sidebar:
    st.header("Tentang")
    st.markdown("""
- **Algoritma:** SVM (SVC)  
- **Dataset:** Pima Indians Diabetes (Kaggle)  
- **Deployment:** Streamlit  
    """)
    st.markdown("---")
    st.caption("Jika error, cek `requirements.txt` dan file model di folder `model/`.")

st.subheader("Input Data Pasien")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
    blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=70, step=1)
    skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20, step=1)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=28.0, step=0.1)
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]], dtype=float)

if st.button("🔍 Prediksi"):
    pred = int(model.predict(input_data)[0])

    # Jika model tidak punya predict_proba (misal probability=False), handle aman
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(input_data)[0][1])

    if pred == 1:
        if proba is not None:
            st.error(f"✅ Hasil: **DIABETES** (probabilitas: {proba:.2f})")
        else:
            st.error("✅ Hasil: **DIABETES**")
    else:
        if proba is not None:
            st.success(f"✅ Hasil: **TIDAK DIABETES** (probabilitas: {proba:.2f})")
        else:
            st.success("✅ Hasil: **TIDAK DIABETES**")

    st.caption("Catatan: Ini hanya model ML berbasis data Kaggle, bukan diagnosis medis.")
