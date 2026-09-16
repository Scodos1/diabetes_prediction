"""
pages/4_Analytics_Dashboard.py
--------------------------------
Module 4: Analytics Dashboard

  - Total patients screened, high-risk count, low-risk count
  - Risk distribution chart
  - Predictions-over-time trend chart
  - Age distribution by risk level
"""

import pandas as pd
import streamlit as st

from db import get_analytics_summary, get_predictions_over_time, get_all_predictions
from style import metric_card

st.title("Analytics Dashboard")
st.markdown(
    '<p class="subtitle">Aggregate view of all screenings recorded in this app.</p>',
    unsafe_allow_html=True,
)

summary = get_analytics_summary()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card("Patients Screened", summary["total_patients_screened"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card("Total Predictions", summary["total_predictions"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card("High Risk", summary["high_risk_count"], variant="high"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card("Low Risk", summary["low_risk_count"], variant="low"), unsafe_allow_html=True)

st.write("")

all_predictions = get_all_predictions()

if not all_predictions:
    st.info("No predictions recorded yet. Run a prediction to populate the dashboard.")
    st.stop()

df = pd.DataFrame(all_predictions)
df["created_at"] = pd.to_datetime(df["created_at"])
df["day"] = df["created_at"].dt.date

st.subheader("Risk Distribution")
risk_counts = df["risk_label"].value_counts().reindex(["HIGH", "LOW"]).fillna(0)
st.bar_chart(
    risk_counts.rename("Count"),
    color="#B23A3A",
)

st.subheader("Predictions Over Time")
trend = get_predictions_over_time()
if trend:
    trend_df = pd.DataFrame(trend).set_index("day")
    trend_df = trend_df.rename(columns={"high_count": "High Risk", "low_count": "Low Risk"})
    st.line_chart(trend_df, color=["#B23A3A", "#2C8050"])

st.subheader("Age Distribution by Risk Level")
age_bins = pd.cut(df["age"], bins=[0, 20, 30, 40, 50, 60, 70, 80, 120],
                   labels=["<20", "20s", "30s", "40s", "50s", "60s", "70s", "80+"])
df["age_bracket"] = age_bins
age_risk = df.groupby(["age_bracket", "risk_label"], observed=True).size().unstack(fill_value=0)
for col in ["HIGH", "LOW"]:
    if col not in age_risk.columns:
        age_risk[col] = 0
age_risk = age_risk[["HIGH", "LOW"]].rename(columns={"HIGH": "High Risk", "LOW": "Low Risk"})
st.bar_chart(age_risk, color=["#B23A3A", "#2C8050"])

st.subheader("Average Glucose Level by Risk Level")
glucose_avg = df.groupby("risk_label", observed=True)["glucose"].mean().reindex(["HIGH", "LOW"])
st.bar_chart(glucose_avg.rename("Avg Glucose (mg/dL)"), color="#0B5FA5")

st.divider()
st.subheader("Export Data")
export_df = df[["name", "gender", "age", "glucose", "bmi", "blood_pressure", "insulin",
                "family_history", "risk_label", "probability_pct", "created_at"]].copy()
export_df.columns = ["Name", "Gender", "Age", "Glucose", "BMI", "Blood Pressure",
                      "Insulin", "Family History", "Risk Label", "Probability %", "Date"]
csv_data = export_df.to_csv(index=False)
st.download_button(
    "Download Predictions as CSV",
    data=csv_data,
    file_name="diabetes_predictions.csv",
    mime="text/csv",
)
