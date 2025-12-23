# app.py (FINAL - siap deploy Streamlit Cloud)
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


st.set_page_config(page_title="SVM Diabetes Predictor", page_icon="🩺", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "svm_model.joblib"
MODEL_DIR.mkdir(exist_ok=True)


@st.cache_resource
def get_or_train_model():
    # 1) Kalau model sudah ada -> load
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # 2) Kalau belum ada -> train dari dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
            "Pastikan file 'diabetes.csv' ada di root project (sejajar dengan app.py)."
        )

    df = pd.read_csv(DATA_PATH)

    # validasi kolom target
    if "Outcome" not in df.columns:
        raise ValueError("Kolom target 'Outcome' tidak ditemukan di diabetes.csv")

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
    ])

    model.fit(X_train, y_train)

    # Simpan supaya run berikutnya tinggal load (lebih cepat)
    joblib.dump(model, MODEL_PATH)

    return model


# Load / Train model dengan error handling
try:
    model = get_or_train_model()
except Exception as e:
    st.error("Aplikasi tidak bisa berjalan karena model/dataset bermasalah ❌")
    st.code(str(e))
    st.stop()


st.title("🩺 Prediksi Diabetes (SVM)")
st.write("Masukkan data pasien, lalu klik **Prediksi** untuk melihat hasil klasifikasi.")

with st.sidebar:
    st.header("Info")
    st.markdown("""
- **Algoritma:** Support Vector Machine (SVC)  
- **Preprocessing:** StandardScaler  
- **Dataset:** Pima Indians Diabetes (Kaggle)  
- **Output:** Prediksi + probabilitas  
    """)
    st.markdown("---")
    st.caption("Jika error saat deploy, pastikan `requirements.txt` dan `diabetes.csv` ada di repo.")


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
    proba = float(model.predict_proba(input_data)[0][1])

    if pred == 1:
        st.error(f"✅ Hasil: **DIABETES** (probabilitas: {proba:.2f})")
    else:
        st.success(f"✅ Hasil: **TIDAK DIABETES** (probabilitas: {proba:.2f})")

    st.caption("Catatan: Ini hanya prediksi berbasis Machine Learning, bukan diagnosis medis.")
