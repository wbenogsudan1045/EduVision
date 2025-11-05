# train_linear_regression.py
"""
Train a Linear Regression model and save outputs
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "student_performance.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna()
print("Dataset preview:")
print(df.head())

FEATURE_COLS = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade']
TARGET_COL = 'FinalGrade'

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("\n=== Linear Regression Results ===")
print(f"Intercept: {model.intercept_:.4f}")
for feature, coef in zip(FEATURE_COLS, model.coef_):
    print(f"  {feature}: {coef:.4f}")

print("\n=== Model Evaluation ===")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

# model_path = os.path.join(MODEL_DIR, "linear_regression_model.pkl")
# with open(model_path, "wb") as f:
#     pickle.dump(model, f)
# with open(os.path.join(MODEL_DIR, "scaler_lr.pkl"), "wb") as f:
#     pickle.dump(scaler, f)

# print(f"\nSaved model → {model_path}")
