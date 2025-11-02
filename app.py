import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "eduvision_secret"

# ===== Directory for model files =====
MODEL_DIR = "models"

# ===== Utility Loaders =====
def load_saved_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        print(f"[WARN] Model file not found: {path}")
        return None
    if name.endswith(".h5"):
        return load_model(path)
    elif name.endswith(".pkl"):
            return pickle.load(f)
    return None

def load_pickle(name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# ===== Load all models =====
models = {
    "linear_regression": load_saved_model("linear_regression_model.pkl"),
    "naive_bayes": load_saved_model("naive_bayes_model.pkl"),
    "knn": load_saved_model("knn_model.pkl"),
    "svm": load_saved_model("svm_model.pkl"),
    "decision_tree": load_saved_model("decision_tree_surrogate.h5"),
    "ann": load_saved_model("ann_model.h5"),
}

# ===== Load scalers, encoders, and feature columns =====
scalers = {
    "linear_regression": load_pickle("scaler_lr.pkl"),
    "naive_bayes": load_pickle("scaler_nb.pkl"),
    "knn": load_pickle("scaler_knn.pkl"),
    "svm": load_pickle("scaler_svm.pkl"),
    "decision_tree": load_pickle("scaler_dt.pkl"),
    "ann": load_pickle("scaler_ann.pkl"),
}

encoders = {
    "linear_regression": None,
    "naive_bayes": load_pickle("label_encoder_nb.pkl"),
    "svm": load_pickle("label_encoder_svm.pkl"),
    "decision_tree": load_pickle("label_encoder_dt.pkl"),
    "knn": load_pickle("label_encoder_knn.pkl"),
    "ann": load_pickle("label_encoder_ann.pkl"),
}

feature_columns = {
    "ann": load_pickle("ann_feature_columns.pkl"),
    "knn": load_pickle("knn_feature_columns.pkl"),
    "decision_tree": load_pickle("dt_feature_columns.pkl"),
}

# ===== Model Input Mapping =====
MODEL_INPUTS = {
    "linear_regression": ["study_hours_per_week", "quiz_average", "attendance_rate"],
    "naive_bayes": ["study_hours_per_week", "quiz_average", "attendance_rate"],
    "knn": ["interest_tags", "preferred_learning_style", "parent_education_level"],
    "svm": ["previous_grades", "quiz_average", "attendance_rate", "final_exam_score"],
    "decision_tree": ["assignment_submission_rate", "quiz_average", "attendance_rate", "final_exam_score"],
    "ann": ["course_enrolled", "interest_tags", "socioeconomic_status"],
}

# ====== ROUTES ======

@app.route('/')
def landing():
    """Landing Page"""
    return render_template("landing.html")

@app.route('/dashboard')
def dashboard():
    """ML Models Dashboard"""
    return render_template("index.html", results={})

@app.route('/predict', methods=["POST"])
def predict():
    """Prediction Logic"""
    results = {}
    model_key = request.form.get("model_name")
    model = models.get(model_key)
    scaler = scalers.get(model_key)
    encoder = encoders.get(model_key)
    cols = feature_columns.get(model_key)

    if model is None:
        results[model_key] = "Error: Model not loaded."
        return render_template("index.html", results=results)

    try:
        input_names = MODEL_INPUTS.get(model_key, [])
        data = {name: request.form.get(f"input_{name}") for name in input_names}

        # Handle categorical models
        if cols is not None:
            df = pd.DataFrame([data]).astype(str)
            df_dummies = pd.get_dummies(df)
            df_dummies = df_dummies.reindex(columns=cols, fill_value=0)
            X = df_dummies.values
        else:
            vals = [float(data[name]) for name in input_names]
            X = np.array(vals).reshape(1, -1)

        if scaler is not None:
            X = scaler.transform(X)

        preds = model.predict(X)
        result = None

        # Regression
        if model_key == "linear_regression":
            result = round(float(preds.flatten()[0]), 2)

        # Multi-class ANN or other classifiers
        elif isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] > 1:
            idx = int(np.argmax(preds, axis=1)[0])
            result = encoder.inverse_transform([idx])[0] if encoder is not None else str(idx)

        # Single-output classifiers
        else:
            idx = int(round(preds.flatten()[0]))
            result = encoder.inverse_transform([idx])[0] if encoder is not None else str(idx)

        results[model_key] = result

    except Exception as e:
        results[model_key] = f"Error: {str(e)}"

    return render_template("index.html", results=results)

# ===== Run App =====
if __name__ == "__main__":
    # For deployment (e.g., Render/Railway), port is provided by environment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
