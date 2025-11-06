import os
import numpy as np
import pickle
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)
app.secret_key = "eduvision_secret_key"

# ===== Helper function =====
def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"[WARNING] Missing: {path}")
    return None

# ===== Load Models =====
models = {
    "linear_regression": load_pickle("models/linear_regression_model.pkl"),
    "naive_bayes": load_pickle("models/naive_bayes_model.pkl"),
    "knn": load_pickle("models/knn_model.pkl"),
    "svm": load_pickle("models/svm_model.pkl"),
    "decision_tree": load_pickle("models/decision_tree_model.pkl"),
    "ann": load_model("models/ann_model.h5") if os.path.exists("models/ann_model.h5") else None
}

# ===== Load Scalers =====
scalers = {
    "linear_regression": load_pickle("models/scaler_lr.pkl"),
    "naive_bayes": load_pickle("models/scaler_nb.pkl"),
    "knn": load_pickle("models/scaler_knn.pkl"),
    "svm": load_pickle("models/scaler_svm.pkl"),
    "decision_tree": load_pickle("models/scaler_dt.pkl"),
    "ann": load_pickle("models/scaler_ann.pkl")
}

# ===== Load Label Encoders =====
label_encoders = {
    "naive_bayes": load_pickle("models/label_encoder_nb.pkl"),
    "knn": load_pickle("models/label_encoder_knn.pkl"),
    "svm": load_pickle("models/label_encoder_svm.pkl"),
    "decision_tree": load_pickle("models/label_encoder_dt.pkl"),
    "ann": load_pickle("models/label_encoder_ann.pkl")
}

# ===== Load Feature Columns =====
feature_columns = {
    "linear_regression": ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade'],
    "naive_bayes": ['Study_Hours_per_Week', 'Attendance_Rate', 'Final_Exam_Score'],
    "knn": load_pickle("models/knn_feature_columns.pkl"),
    "svm": ['Attendance', 'PreviousScores', 'Parental_Involvement'],
    "decision_tree": load_pickle("models/dt_feature_columns.pkl"),
    "ann": load_pickle("models/ann_feature_columns.pkl")
}

# ===== Home Page =====
@app.route("/")
def index():
    return render_template("landing.html")
    

# ===== Linear Regression =====
@app.route("/linear_regression", methods=["GET", "POST"])
def linear_regression():
    result = None
    features = feature_columns["linear_regression"]
    if request.method == "POST":
        try:
            X = np.array([[float(request.form.get(f"input_{f}", 0)) for f in features]])
            if scalers["linear_regression"]:
                X = scalers["linear_regression"].transform(X)
            prediction = models["linear_regression"].predict(X)
            result = round(float(prediction[0]), 2)
        except Exception as e:
            result = f"Error: {e}"
    return render_template("linear_regression.html", result=result, feature_columns=features)

# ===== Naive Bayes =====
@app.route("/naive_bayes", methods=["GET", "POST"])
def naive_bayes():
    result = None
    features = feature_columns["naive_bayes"]
    if request.method == "POST":
        try:
            X = np.array([[float(request.form.get(f"input_{f}", 0)) for f in features]])
            if scalers["naive_bayes"]:
                X = scalers["naive_bayes"].transform(X)
            prediction = models["naive_bayes"].predict(X)
            encoder = label_encoders["naive_bayes"]
            result = encoder.inverse_transform(prediction)[0] if encoder else prediction[0]
        except Exception as e:
            result = f"Error: {e}"
    return render_template("naive_bayes.html", result=result, feature_columns=features)

# ===== KNN =====
@app.route("/knn", methods=["GET", "POST"])
def knn():
    result = None
    feature_cols = feature_columns.get("knn", [])
    if request.method == "POST":
        try:
            input_data = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
            raw_inputs = {f: request.form.get(f"input_{f}", "") for f in ['ParentalInvolvement', 'SchoolType', 'MotivationLevel']}
            input_features = pd.get_dummies(pd.DataFrame([raw_inputs]))
            for col in feature_cols:
                input_data[col] = input_features.iloc[0].get(col, 0)
            X = input_data.values
            if scalers["knn"]:
                X = scalers["knn"].transform(X)
            prediction = models["knn"].predict(X)
            encoder = label_encoders["knn"]
            result = encoder.inverse_transform(prediction)[0] if encoder else prediction[0]
        except Exception as e:
            result = f"Error: {e}"
    return render_template("knn.html", result=result, feature_columns=feature_cols)



@app.route("/svm", methods=["GET", "POST"])
def svm():
    result = None
    svm_feature_cols = load_pickle("models/svm_feature_columns.pkl")
    if request.method == "POST":
        try:
            raw_inputs = {
                "Attendance": float(request.form.get("input_Attendance", 0)),
                "PreviousScores": float(request.form.get("input_PreviousScores", 0)),
                "Parental_Involvement": request.form.get("input_Parental_Involvement", "")
            }
            X_df = pd.DataFrame([raw_inputs])
            X_dummies = pd.get_dummies(X_df)
            X_dummies = X_dummies.reindex(columns=svm_feature_cols, fill_value=0)
            
            print("Form data:", raw_inputs)
            print("X_df:\n", X_df)
            print("Dummies:\n", X_dummies)
            
            if scalers["svm"]:
                X = scalers["svm"].transform(X_dummies)
            else:
                X = X_dummies.values
            prediction = models["svm"].predict(X)
            encoder = label_encoders["svm"]
            result = encoder.inverse_transform(prediction)[0] if encoder else prediction[0]
        except Exception as e:
            result = f"Error: {e}"
    return render_template("svm.html", result=result, feature_columns=feature_columns["svm"])





# ===== Decision Tree =====
@app.route("/decision_tree", methods=["GET", "POST"])
def decision_tree():
    result = None
    feature_cols = ['Numerical Aptitude', 'Spatial Aptitude', 'Perceptual Aptitude', 'Abstract Reasoning', 'Verbal Reasoning']
    
    if request.method == "POST":
        try:
            # Direct numeric input reading (use feature_cols to pull each field)
            X = np.array([[float(request.form.get(f"input_{f}", 0)) for f in feature_cols]])
            # Scale
            if scalers["decision_tree"]:
                X = scalers["decision_tree"].transform(X)
            # Predict
            prediction = models["decision_tree"].predict(X)
            # Decode prediction
            encoder = label_encoders["decision_tree"]
            result = encoder.inverse_transform(prediction)[0] if encoder else str(prediction[0])
        except Exception as e:
            result = f"Error: {e}"
    
    return render_template("decision_tree.html", result=result, feature_columns=feature_cols)


# ===== ANN =====
@app.route("/ann", methods=["GET", "POST"])
def ann():
    result = None
    feature_cols = ['math score', 'reading score', 'writing score']
    
    if request.method == "POST":
        try:
            X = np.array([[float(request.form.get(f"input_{f.replace(' ', '_')}", 0)) for f in feature_cols]])
            if scalers["ann"]:
                X = scalers["ann"].transform(X)
            prediction = models["ann"].predict(X)
            label_encoder = label_encoders["ann"]
            predicted_class = np.argmax(prediction)
            
            # Map class to readable string
            class_map = {0: "Not Achiever", 1: "Achiever"}
            result = class_map.get(predicted_class, str(predicted_class))
            
        except Exception as e:
            result = f"Error: {e}"
    
    return render_template("ann.html", result=result, feature_columns=feature_cols)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
