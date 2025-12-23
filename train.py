import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Path
DATA_PATH = "diabetes.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split data
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline SVM
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=42))
])

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_PATH)

print("✅ Model berhasil dibuat di:", MODEL_PATH)
