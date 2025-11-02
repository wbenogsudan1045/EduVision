# train_decision_tree_surrogate.py
"""
Train a small NN surrogate for Decision Tree classification and save .h5
Saves:
 - models/decision_tree_surrogate.h5
 - models/scaler_dt.pkl
 - models/label_encoder_dt.pkl
 - models/dt_feature_columns.pkl
"""
import os
import pickle
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "eduvision_dataset.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna()

TARGET_COL = "dropout_likelihood"
FEATURES = [
    "assignment_submission_rate",
    "quiz_average",
    "attendance_rate",
    "final_exam_score"
]

missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

X_df = df[FEATURES]
y_raw = df[TARGET_COL].astype(str)

le = LabelEncoder()
y_enc = le.fit_transform(y_raw)
y_cat = to_categorical(y_enc)

# If any categorical in X_df, one-hot them (here all numeric; still safe)
X_dummies = pd.get_dummies(X_df)
feature_cols = list(X_dummies.columns)
with open(os.path.join(MODEL_DIR, "dt_feature_columns.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)

X = X_dummies.values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=2)

model.save(os.path.join(MODEL_DIR, "decision_tree_surrogate.h5"))
with open(os.path.join(MODEL_DIR, "label_encoder_dt.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(MODEL_DIR, "scaler_dt.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("✅ Saved decision_tree_surrogate.h5 and artifacts.")
