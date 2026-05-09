"""
app.py  –  Flask backend for Loan Approval Prediction System
Run:
    python backend/app.py
API available at http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, "model", "loan_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "model", "encoders.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_names.pkl")

try:
    model         = joblib.load(MODEL_PATH)
    encoders      = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model files not found. Run  python model/train_model.py  first.")
    model = encoders = feature_names = None


# ── Helper ────────────────────────────────────────────────────────────────────
def preprocess_input(data: dict) -> pd.DataFrame:
    """Convert raw form data into model-ready DataFrame."""
    df = pd.DataFrame([data])

    cat_cols = ["Gender", "Married", "Dependents", "Education",
                "Self_Employed", "Property_Area"]

    for col in cat_cols:
        le = encoders.get(col)
        if le is None:
            raise ValueError(f"No encoder found for column: {col}")
        val = df[col].astype(str).values[0]
        if val not in le.classes_:
            # fallback to most-common class
            val = le.classes_[0]
        df[col] = le.transform([val])

    # Numeric casts
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome",
                    "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df[feature_names]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    body = request.get_json(force=True)
    if not body:
        return jsonify({"error": "No JSON body received"}), 400

    required_fields = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area"
    ]
    missing = [f for f in required_fields if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = preprocess_input(body)
        prediction    = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0].tolist()
        confidence    = round(max(probabilities) * 100, 2)

        # Risk level
        approval_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        if approval_prob >= 0.75:
            risk = "Low Risk"
        elif approval_prob >= 0.50:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        return jsonify({
            "prediction":   prediction,
            "status":       "Approved" if prediction == 1 else "Rejected",
            "confidence":   confidence,
            "risk_level":   risk,
            "approval_probability": round(approval_prob * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
