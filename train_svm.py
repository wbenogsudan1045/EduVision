# train_svm.py
"""
Train a Support Vector Machine (SVM) classifier and save artifacts
Predict Motivation_Level from StudentPerformanceFactors.csv
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), "SVM_motivation_dataset.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and preview data
df = pd.read_csv(DATA_PATH).dropna()
print("Dataset preview:")
print(df.head())

# Feature columns and target choice
FEATURE_COLS = ['Attendance', 'PreviousScores', 'Parental_Involvement']
TARGET_COL = 'MotivationLevel'

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Encode categorical features (includes School_Type, Peer_Influence, etc.)
X = pd.get_dummies(df[FEATURE_COLS])
y_raw = df[TARGET_COL].astype(str)

le = LabelEncoder()
y = le.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Right after you create X = pd.get_dummies(df[FEATURE_COLS]):
X = pd.get_dummies(df[FEATURE_COLS])
feature_cols = list(X.columns)
with open(os.path.join(MODEL_DIR, "svm_feature_columns.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)


model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_val, y_pred)

print("\n=== SVM Model Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nDetailed Classification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# Save artifacts if needed
with open(os.path.join(MODEL_DIR, "svm_model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "scaler_svm.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, "label_encoder_svm.pkl"), "wb") as f:
    pickle.dump(le, f)

print("Saved SVM artifacts.")
