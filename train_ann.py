"""
Train a deeper ANN for personalized learning style prediction and save as ann_model.h5
Saves:
 - models/ann_model.h5
 - models/label_encoder_ann.pkl
 - models/scaler_ann.pkl
 - models/ann_feature_columns.pkl
"""
import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# === File Paths ===
DATA_PATH = os.path.join(os.path.dirname(__file__), "StudentsPerformance_with_high_achiever.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# === Load dataset ===
df = pd.read_csv(DATA_PATH).dropna()


# === Specify dependent and independent variables ===
TARGET_COL = "high_achiever"
FEATURES = [
    "math score",
    "reading score",
    "writing score"
]


# Safety: verify columns exist
missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")


# === Split X and y ===
# Keep numeric features as float (not string)
X_df = df[FEATURES].astype(float)
y_raw = df[TARGET_COL].astype(str)


# === Encode target (categorical) ===
le = LabelEncoder()
y_enc = le.fit_transform(y_raw)
y_cat = to_categorical(y_enc)


# === Feature columns for numeric features are direct names ===
feature_cols = FEATURES
with open(os.path.join(MODEL_DIR, "ann_feature_columns.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)


X = X_df.values


# === Scale numeric features ===
scaler = StandardScaler()
X = scaler.fit_transform(X)


# === Train-validation split (stratified for balanced labels) ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
)


# === Compute class weights for imbalance ===
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_enc), y=y_enc
)
class_weight_dict = dict(enumerate(class_weights))


# === Build ANN model ===
model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(y_cat.shape[1], activation="softmax")
])


# === Compile & Train ===
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=32,
    verbose=2,
    class_weight=class_weight_dict
)


# === Findings ===
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]


print("\n=== Model Findings ===")
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Final Training Loss: {train_loss:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")


# === Save model and encoders/scaler ===
model.save(os.path.join(MODEL_DIR, "ann_model.h5"))
with open(os.path.join(MODEL_DIR, "label_encoder_ann.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(MODEL_DIR, "scaler_ann.pkl"), "wb") as f:
    pickle.dump(scaler, f)


print("✅ Saved ann_model.h5, encoders/scalers, and ann_feature_columns.pkl")
