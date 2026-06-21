"""
pages/0_Home.py
----------------
Landing page: quick overview + at-a-glance stats pulled from the
Analytics module, with shortcuts into each module.
"""

import streamlit as st

from db import get_analytics_summary
from style import metric_card

st.title("Diabetes Risk Predictor")
st.markdown(
    '<p class="subtitle">TabNet-powered risk screening, patient records, '
    'and reporting in one place.</p>',
    unsafe_allow_html=True,
)

summary = get_analytics_summary()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        metric_card("Patients Screened", summary["total_patients_screened"]),
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        metric_card("High Risk", summary["high_risk_count"], variant="high"),
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        metric_card("Low Risk", summary["low_risk_count"], variant="low"),
        unsafe_allow_html=True,
    )

st.write("")
st.write("")

st.subheader("Get started")

c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/1_Patient_Registration.py", label="Register a new patient", icon="📝")
    st.page_link("pages/3_History.py", label="View patient history", icon="📂")
with c2:
    st.page_link("pages/2_Prediction.py", label="Run a risk prediction", icon="🩺")
    st.page_link("pages/4_Analytics_Dashboard.py", label="Open the analytics dashboard", icon="📊")

st.page_link("pages/5_Results.py", label="View model results & confusion matrix", icon="🧪")

st.write("")
st.caption(
    "This tool provides statistical risk estimates from a machine learning model "
    "and is not a substitute for professional medical diagnosis."
)
