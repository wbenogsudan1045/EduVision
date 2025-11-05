# train_naive_bayes.py
"""
Train a Naive Bayes model on eduvision_dataset.csv and save model artifacts
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), "student_performance_dataset.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna()
print("Dataset preview:")
print(df.head())

FEATURE_COLS = ['Study_Hours_per_Week', 'Attendance_Rate', 'Final_Exam_Score']
TARGET_COL = 'Pass_Fail'

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

X = df[FEATURE_COLS].values
y_raw = df[TARGET_COL].astype(str)

le = LabelEncoder()
y = le.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_val, y_pred)

print("\n=== Naive Bayes Model Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nDetailed Classification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

with open(os.path.join(MODEL_DIR, "naive_bayes_model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "scaler_nb.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, "label_encoder_nb.pkl"), "wb") as f:
    pickle.dump(le, f)

print("Saved Naive Bayes artifacts.")
