"""
pages/1_Patient_Registration.py
--------------------------------
Module 1: Patient Registration

Captures Name, Date of Birth, Gender, Phone Number and stores them in
the patients table. Age is derived from Date of Birth wherever the
model or display needs it, so it stays accurate over time instead of
going stale. Registered patients then become selectable in the
Prediction module (Module 2).
"""

import re
from datetime import date

import streamlit as st

from db import register_patient, list_patients, calculate_age

PHONE_REGEX = re.compile(r"^[\d\-\+\(\)\s]{7,20}$")

st.title("Patient Registration")
st.markdown(
    '<p class="subtitle">Add a new patient before running a risk prediction.</p>',
    unsafe_allow_html=True,
)

with st.form("registration_form", clear_on_submit=True):
    name = st.text_input("Name")
    col1, col2 = st.columns(2)
    with col1:
        date_of_birth = st.date_input(
            "Date of Birth",
            value=date(1990, 1, 1),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
        )
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    phone_number = st.text_input("Phone Number", placeholder="e.g. 555-0101")

    submitted = st.form_submit_button("Register Patient")

if submitted:
    if not name.strip():
        st.error("Please enter the patient's name.")
    elif not phone_number.strip():
        st.error("Please enter a phone number.")
    elif not PHONE_REGEX.match(phone_number.strip()):
        st.error("Please enter a valid phone number (7-20 digits, spaces, or dashes).")
    else:
        clean_name = " ".join(name.split())
        clean_phone = phone_number.strip()
        patient_id = register_patient(clean_name, date_of_birth.isoformat(), gender, clean_phone)
        st.success(f"Registered **{clean_name}** (Patient ID: {patient_id}).")
        st.page_link(
            "pages/2_Prediction.py",
            label="Run a risk prediction for this patient →",
            icon="🩺",
        )

st.divider()
st.subheader("Registered Patients")

search_text = st.text_input("Search by name or phone number", key="patient_search")
patients = list_patients(search_text)

if not patients:
    st.info("No patients registered yet." if not search_text else "No matching patients found.")
else:
    st.dataframe(
        [
            {
                "Patient ID": p["patient_id"],
                "Name": p["name"],
                "Date of Birth": p["date_of_birth"],
                "Age": calculate_age(p["date_of_birth"]),
                "Gender": p["gender"],
                "Phone Number": p["phone_number"],
                "Registered": p["created_at"][:10],
            }
            for p in patients
        ],
        use_container_width=True,
        hide_index=True,
    )
