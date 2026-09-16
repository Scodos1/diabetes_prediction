"""
pages/3_History.py
-------------------
Module 3: History Module

  - Previous predictions, listed newest first
  - Search patient records by name or phone number
  - View/download a PDF report for any past prediction
  - Export all results as CSV
"""

import pandas as pd
import streamlit as st

from db import search_history, get_patient, get_prediction
from report import build_patient_report_pdf
from style import risk_pill

st.title("History")
st.markdown(
    '<p class="subtitle">Previous predictions and patient record search.</p>',
    unsafe_allow_html=True,
)

search_text = st.text_input("Search patient records (name or phone number)")
records = search_history(search_text)

if not records:
    st.info("No prediction history yet." if not search_text else "No matching records found.")
    st.stop()

st.caption(f"{len(records)} record(s) found.")

for r in records:
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            st.markdown(f"**{r['name']}**  ·  {r['gender']}  ·  {r['phone_number']}")
            st.caption(f"Recorded {r['created_at'][:16].replace('T', ' ')}")
        with col2:
            st.markdown(risk_pill(r["risk_label"]), unsafe_allow_html=True)
            st.caption(f"Probability: {r['probability_pct']}%")
        with col3:
            st.caption(
                f"Glucose {r['glucose']} · BMI {r['bmi']} · BP {r['blood_pressure']} · "
                f"Insulin {r['insulin']} · Age {r['age']} · "
                f"Family history: {'Yes' if r['family_history'] else 'No'}"
            )

        patient = get_patient(r["patient_id"])
        prediction = get_prediction(r["prediction_id"])
        if patient and prediction:
            pdf_bytes = build_patient_report_pdf(patient, prediction)
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"diabetes_risk_report_{patient['name'].replace(' ', '_')}_{r['prediction_id']}.pdf",
                mime="application/pdf",
                key=f"dl_{r['prediction_id']}",
            )

if records:
    st.divider()
    export_df = pd.DataFrame([
        {
            "Patient": r["name"],
            "Gender": r["gender"],
            "Phone": r["phone_number"],
            "Age": r["age"],
            "Glucose": r["glucose"],
            "BMI": r["bmi"],
            "Blood Pressure": r["blood_pressure"],
            "Insulin": r["insulin"],
            "Family History": "Yes" if r["family_history"] else "No",
            "Risk": r["risk_label"],
            "Probability %": r["probability_pct"],
            "Date": r["created_at"][:10],
        }
        for r in records
    ])
    st.download_button(
        "Export All Results as CSV",
        data=export_df.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv",
    )
