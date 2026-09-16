"""
pages/2_Prediction.py
----------------------
Module 2: Prediction Module

The original MVP flow, now attached to a registered patient:
  Select Patient -> Enter Health Data -> Click Predict -> AI Analyzes Data
  -> Show Risk Result -> (optionally) Generate PDF report

Saves every prediction to the database so it shows up in the History
Module (Module 3) and Analytics Dashboard (Module 4).
"""

import streamlit as st

from predict import DiabetesRiskPredictor
from db import list_patients, save_prediction, get_patient, calculate_age
from report import build_patient_report_pdf

st.title("Prediction")
st.markdown(
    '<p class="subtitle">Current AI prediction for a registered patient, '
    "using the TabNet risk model.</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    return DiabetesRiskPredictor()


try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

patients = list_patients()
if not patients:
    st.warning("No patients registered yet.")
    st.page_link("pages/1_Patient_Registration.py", label="Register a patient first →", icon="📝")
    st.stop()

patient_options = {f'{p["name"]} (ID {p["patient_id"]}, {p["phone_number"]})': p["patient_id"] for p in patients}
selected_label = st.selectbox("Patient", list(patient_options.keys()))
selected_patient_id = patient_options[selected_label]
patient = get_patient(selected_patient_id)
age = calculate_age(patient["date_of_birth"])

st.caption(f"Date of Birth: {patient['date_of_birth']}  ·  Age: {age}")

# ---------------------------------------------------------------------------
# 1. Input Form (with session state persistence)
# ---------------------------------------------------------------------------
for key, default in [("glucose", 110.0), ("bmi", 25.0), ("bp", 80.0), ("insulin", 85.0), ("fh", "No")]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        glucose = st.number_input(
            "Glucose Level (mg/dL)", min_value=0.0, max_value=400.0,
            value=st.session_state.glucose, step=1.0,
        )
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=80.0,
            value=st.session_state.bmi, step=0.1,
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0.0, max_value=250.0,
            value=st.session_state.bp, step=1.0,
        )

    with col2:
        insulin = st.number_input(
            "Insulin Level (mu U/ml)", min_value=0.0, max_value=900.0,
            value=st.session_state.insulin, step=1.0,
        )
        family_history = st.radio(
            "Family History of Diabetes", ["No", "Yes"], horizontal=True,
            index=0 if st.session_state.fh == "No" else 1,
        )

    # 2. Predict Button
    submitted = st.form_submit_button("Predict Risk")

# ---------------------------------------------------------------------------
# 3. Result Screen
# ---------------------------------------------------------------------------
if submitted:
    st.session_state.glucose = glucose
    st.session_state.bmi = bmi
    st.session_state.bp = blood_pressure
    st.session_state.insulin = insulin
    st.session_state.fh = family_history

    warnings = []
    if glucose < 40 or glucose > 400:
        warnings.append(f"Glucose level ({glucose} mg/dL) is outside typical clinical range (40-400).")
    if bmi < 10 or bmi > 80:
        warnings.append(f"BMI ({bmi}) is outside typical clinical range (10-80).")
    if blood_pressure < 30 or blood_pressure > 250:
        warnings.append(f"Blood pressure ({blood_pressure} mm Hg) is outside typical clinical range (30-250).")
    if insulin < 2 or insulin > 900:
        warnings.append(f"Insulin level ({insulin} mu U/ml) is outside typical clinical range (2-900).")
    if glucose == 0 or blood_pressure == 0 or bmi == 0 or insulin == 0:
        warnings.append("A value of 0 is treated as missing data by the model and may produce unreliable results.")
    if warnings:
        for w in warnings:
            st.warning(w)

    with st.spinner("AI is analyzing your data..."):
        result = predictor.predict_risk(
            age=age,
            glucose=glucose,
            bmi=bmi,
            blood_pressure=blood_pressure,
            insulin=insulin,
            family_history=(family_history == "Yes"),
        )

    inputs = dict(
        age=age,
        glucose=glucose,
        bmi=bmi,
        blood_pressure=blood_pressure,
        insulin=insulin,
        family_history=(family_history == "Yes"),
    )
    prediction_id = save_prediction(selected_patient_id, inputs, result)

    card_class = "result-high" if result["label"] == "HIGH" else "result-low"
    value_class = "risk-value-high" if result["label"] == "HIGH" else "risk-value-low"

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="risk-label">Diabetes Risk</div>
            <div class="{value_class}">{result['label']}</div>
            <div class="prob-value">Probability: {result['probability_pct']}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What influenced this prediction?"):
        explanation = predictor.explain(
            age=age,
            glucose=glucose,
            bmi=bmi,
            blood_pressure=blood_pressure,
            insulin=insulin,
            family_history=(family_history == "Yes"),
        )
        sorted_features = sorted(explanation.items(), key=lambda kv: kv[1], reverse=True)
        for feat, weight in sorted_features:
            st.write(f"**{feat}**")
            st.progress(min(max(weight, 0.0), 1.0))

    # --- Module 5: Report Generation ---
    pdf_prediction = {
        "prediction_id": prediction_id,
        "age": age,
        "glucose": glucose,
        "bmi": bmi,
        "blood_pressure": blood_pressure,
        "insulin": insulin,
        "family_history": family_history == "Yes",
        "risk_label": result["label"],
        "probability_pct": result["probability_pct"],
    }
    pdf_bytes = build_patient_report_pdf(patient, pdf_prediction)
    st.download_button(
        "Download PDF Report",
        data=pdf_bytes,
        file_name=f"diabetes_risk_report_{patient['name'].replace(' ', '_')}.pdf",
        mime="application/pdf",
    )

    st.caption(
        "This tool provides a statistical risk estimate based on a machine learning "
        "model and is not a medical diagnosis. Please consult a healthcare "
        "professional for clinical evaluation."
    )
