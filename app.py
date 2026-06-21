"""
app.py
------
Entry point. Sets up Streamlit's native multipage navigation across
the 5 modules and initializes the database on first run.

Run with:
    streamlit run app.py
"""

import streamlit as st

from db import init_db
from style import inject_style

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

init_db()
inject_style()

home_page = st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True)
registration_page = st.Page("pages/1_Patient_Registration.py", title="Patient Registration", icon="📝")
prediction_page = st.Page("pages/2_Prediction.py", title="Prediction", icon="🩺")
history_page = st.Page("pages/3_History.py", title="History", icon="📂")
analytics_page = st.Page("pages/4_Analytics_Dashboard.py", title="Analytics Dashboard", icon="📊")
results_page = st.Page("pages/5_Results.py", title="Model Results", icon="🧪")

nav = st.navigation(
    {
        "Diabetes Risk Predictor": [home_page],
        "Modules": [registration_page, prediction_page, history_page, analytics_page],
        "Model": [results_page],
    }
)
nav.run()
