# train_knn.py
"""
Train a KNN classifier and save artifacts:
 - models/knn_model.pkl
 - models/scaler_knn.pkl
 - models/label_encoder_knn.pkl
 - models/knn_feature_columns.pkl
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = os.path.join(os.path.dirname(__file__), "KNN_peerinfluence_overlap.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna()
print("Dataset preview:")
print(df.head())

FEATURE_COLS = ['ParentalInvolvement', 'SchoolType', 'MotivationLevel']
TARGET_COL = 'PeerInfluence'

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

X_df = df[FEATURE_COLS].astype(str)
X_dummies = pd.get_dummies(X_df)
feature_cols = list(X_dummies.columns)
with open(os.path.join(MODEL_DIR, "knn_feature_columns.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)

X = X_dummies.values

y_raw = df[TARGET_COL].astype(str)
le = LabelEncoder()
y = le.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_val, y_pred)

print("\n=== KNN Model Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nDetailed Classification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

model_path = os.path.join(MODEL_DIR, "knn_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "scaler_knn.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, "label_encoder_knn.pkl"), "wb") as f:
    pickle.dump(le, f)

print(f"\nSaved model → {model_path}")
print("Saved scaler and encoder and knn_feature_columns.pkl")
