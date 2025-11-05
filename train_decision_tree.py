"""
Train a Decision Tree classifier for strand prediction and save artifacts.
Saves:
 - models/decision_tree_model.pkl
 - models/scaler_dt.pkl
 - models/label_encoder_dt.pkl
 - models/dt_feature_columns.pkl
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# === Setup paths ===
DATA_PATH = os.path.join(os.path.dirname(__file__), "Data_high_separation_strands.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load data ===
df = pd.read_csv(DATA_PATH).dropna()

TARGET_COL = "StrandCategory"
FEATURES = ['Numerical Aptitude', 'Spatial Aptitude', 'Perceptual Aptitude', 'Abstract Reasoning', 'Verbal Reasoning']

# === Check for missing columns ===
missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# === Split X and y ===
X_df = df[FEATURES]
y_raw = df[TARGET_COL]

# === Convert continuous target (if any) to categorical ===
if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 10:
    y_raw = pd.cut(
        y_raw,
        bins=[-float("inf"), 0.33, 0.66, float("inf")],
        labels=["Low", "Medium", "High"]
    )
    print("⚙️ Target was numeric — converted to categories: Low, Medium, High")

# === Encode target ===
le = LabelEncoder()
y_enc = le.fit_transform(y_raw)

# === Encode features (dummy encoding if necessary) ===
X_dummies = pd.get_dummies(X_df)
feature_cols = list(X_dummies.columns)
with open(os.path.join(MODEL_DIR, "dt_feature_columns.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)

X = X_dummies.values

# === Scale numeric features ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === Train/validation split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# === Define Decision Tree ===
model = DecisionTreeClassifier(
    criterion="entropy",     # or "gini"
    max_depth=6,             # can tune this
    min_samples_split=4,
    random_state=42
)

# === Train the model ===
model.fit(X_train, y_train)

# === Evaluate Model ===
y_val_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_val, y_val_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n--- Confusion Matrix ---")
print(cm)

print("\n--- Classification Report ---")
print(classification_report(y_val, y_val_pred, target_names=le.classes_, zero_division=0))

# === (Optional) Show Tree Structure ===
tree_rules = export_text(model, feature_names=feature_cols)
print("\n🌳 Decision Tree Rules:\n")
print(tree_rules)

# === Save Artifacts ===
with open(os.path.join(MODEL_DIR, "decision_tree_model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "label_encoder_dt.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(MODEL_DIR, "scaler_dt.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Saved decision_tree_model.pkl and artifacts.")
