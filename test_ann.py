import numpy as np
import pandas as pd
import pickle

# Load training-time dummy column names (IMPORTANT for alignment)
svm_feature_cols = pickle.load(open("models/svm_feature_columns.pkl", "rb"))

# Load model artifacts
model = pickle.load(open("models/svm_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler_svm.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder_svm.pkl", "rb"))

# Sample input as a dict (just like your app/html form)
sample_input = {
    "Attendance": 100,
    "PreviousScores": 100,
    "Parental_Involvement": "high"
}
# Wrap in a DataFrame
sample_df = pd.DataFrame([sample_input])

# One-hot encode and align with training columns
sample_dummies = pd.get_dummies(sample_df)
sample_aligned = sample_dummies.reindex(columns=svm_feature_cols, fill_value=0)

# Scale as in training
sample_scaled = scaler.transform(sample_aligned)

# Predict
pred_class = model.predict(sample_scaled)[0]
print("Predicted class index:", pred_class)
print("Predicted label:", label_encoder.inverse_transform([pred_class])[0])
