"""
train_model.py
--------------
Trains the Loan Approval Prediction model and saves it to model/loan_model.pkl
Run this ONCE before starting the backend server:
    python model/train_model.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# ── 1. Generate / load dataset ────────────────────────────────────────────────
# If you have a real CSV (e.g. from Kaggle), replace this block with:
#   df = pd.read_csv("data/loan_data.csv")

np.random.seed(42)
n = 800

data = {
    "Loan_ID": [f"LP{str(i).zfill(6)}" for i in range(n)],
    "Gender":           np.random.choice(["Male", "Female"], n, p=[0.8, 0.2]),
    "Married":          np.random.choice(["Yes", "No"],      n, p=[0.65, 0.35]),
    "Dependents":       np.random.choice(["0", "1", "2", "3+"], n),
    "Education":        np.random.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22]),
    "Self_Employed":    np.random.choice(["Yes", "No"],      n, p=[0.14, 0.86]),
    "ApplicantIncome":  np.random.randint(1500, 15000, n),
    "CoapplicantIncome":np.random.choice([0, 1000, 2000, 3000, 4000], n),
    "LoanAmount":       np.random.randint(50, 700, n),
    "Loan_Amount_Term": np.random.choice([120, 180, 240, 360, 480], n),
    "Credit_History":   np.random.choice([0, 1], n, p=[0.15, 0.85]),
    "Property_Area":    np.random.choice(["Urban", "Semiurban", "Rural"], n),
}
df = pd.DataFrame(data)

# Target: biased toward approved when credit_history=1 and income is decent
df["Loan_Status"] = (
    (df["Credit_History"] == 1) &
    (df["ApplicantIncome"] > 3000) &
    (df["LoanAmount"] < 400)
).astype(int)
# Add noise
flip_idx = np.random.choice(df.index, size=int(0.12 * n), replace=False)
df.loc[flip_idx, "Loan_Status"] = 1 - df.loc[flip_idx, "Loan_Status"]

# ── 2. Save raw dataset ───────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/loan_data.csv", index=False)
print(f"Dataset saved → data/loan_data.csv  ({len(df)} rows)")

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
df_model = df.drop("Loan_ID", axis=1).copy()

# Fill any NaN (for real datasets)
df_model["Gender"].fillna("Male", inplace=True)
df_model["Married"].fillna("No", inplace=True)
df_model["Dependents"].fillna("0", inplace=True)
df_model["Self_Employed"].fillna("No", inplace=True)
df_model["LoanAmount"].fillna(df_model["LoanAmount"].median(), inplace=True)
df_model["Credit_History"].fillna(1, inplace=True)

# Encode categoricals
le = LabelEncoder()
cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
encoders = {}
for col in cat_cols:
    le_col = LabelEncoder()
    df_model[col] = le_col.fit_transform(df_model[col].astype(str))
    encoders[col] = le_col

# ── 4. Train / test split ─────────────────────────────────────────────────────
X = df_model.drop("Loan_Status", axis=1)
y = df_model["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_names = list(X.columns)

# ── 5. Train models & pick best ───────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

best_model = None
best_acc   = 0
results    = {}

print("\n── Model Evaluation ──────────────────────────")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    print(f"{name:25s}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
    if acc > best_acc:
        best_acc   = acc
        best_model = model
        best_name  = name

print(f"\nBest model → {best_name}  (accuracy={best_acc:.3f})")

# ── 6. Save artifacts ─────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(best_model,    "model/loan_model.pkl")
joblib.dump(encoders,      "model/encoders.pkl")
joblib.dump(feature_names, "model/feature_names.pkl")
print("Saved → model/loan_model.pkl, model/encoders.pkl, model/feature_names.pkl")
