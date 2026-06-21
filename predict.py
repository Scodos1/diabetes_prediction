"""
predict.py
----------
Inference helper used by app.py. Loads the trained TabNet model and
scaler once, then exposes a single predict_risk() function that takes
the 6 MVP form inputs and returns a HIGH/LOW risk label plus
probability, matching the MVP Result Screen spec:

    Diabetes Risk: HIGH
    Probability: 82%
"""

import os
import joblib
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

from data import MVP_FEATURES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "tabnet_diabetes.zip")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

RISK_THRESHOLD = 0.5  # probability >= threshold => HIGH risk


class DiabetesRiskPredictor:
    """Thin wrapper around the trained TabNet model + scaler.

    Loaded once per Streamlit session (see app.py's @st.cache_resource)
    so repeated predictions don't reload the model from disk.
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}.\n"
                "Run `python train_model.py` first to train and save the "
                "TabNet model."
            )
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"No fitted scaler found at {SCALER_PATH}.\n"
                "Run `python train_model.py` first."
            )

        self.model = TabNetClassifier()
        self.model.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def predict_risk(
        self,
        age: float,
        glucose: float,
        bmi: float,
        blood_pressure: float,
        insulin: float,
        family_history: bool,
    ) -> dict:
        """Run the 6 MVP inputs through the trained TabNet model.

        Returns:
            {
                "label": "HIGH" | "LOW",
                "probability": float in [0, 1],   # probability of HIGH risk
                "probability_pct": int,            # rounded percentage, 0-100
            }
        """
        # Feature order MUST match MVP_FEATURES from data.py exactly,
        # since that is the order the model was trained on.
        row = {
            "Age": age,
            "Glucose": glucose,
            "BMI": bmi,
            "BloodPressure": blood_pressure,
            "Insulin": insulin,
            "FamilyHistory": 1 if family_history else 0,
        }
        x = np.array([[row[feat] for feat in MVP_FEATURES]], dtype=float)
        x_scaled = self.scaler.transform(x)

        proba_high = float(self.model.predict_proba(x_scaled)[0, 1])
        label = "HIGH" if proba_high >= RISK_THRESHOLD else "LOW"

        # For LOW risk, the MVP spec shows the LOW-class probability
        # (e.g. "Diabetes Risk: LOW / Probability: 18%"), so we report
        # whichever class was predicted.
        display_proba = proba_high if label == "HIGH" else (1 - proba_high)

        return {
            "label": label,
            "probability": display_proba,
            "probability_pct": round(display_proba * 100),
            "raw_high_risk_probability": proba_high,
        }

    def explain(self, age, glucose, bmi, blood_pressure, insulin, family_history):
        """Return TabNet's built-in feature attribution mask for this
        single prediction (TabNet's interpretability advantage,
        referenced in Section 2.4 / 3.10 of the methodology as a key
        reason TabNet was chosen over a black-box model).
        """
        row = {
            "Age": age,
            "Glucose": glucose,
            "BMI": bmi,
            "BloodPressure": blood_pressure,
            "Insulin": insulin,
            "FamilyHistory": 1 if family_history else 0,
        }
        x = np.array([[row[feat] for feat in MVP_FEATURES]], dtype=float)
        x_scaled = self.scaler.transform(x)
        explain_matrix, _ = self.model.explain(x_scaled)
        importances = explain_matrix[0]
        total = importances.sum() or 1.0
        return {
            feat: float(val / total) for feat, val in zip(MVP_FEATURES, importances)
        }
