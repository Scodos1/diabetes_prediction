"""
report.py
---------
Module 5: Report Generation.

Builds a one-page clinical-style PDF summary for a single prediction,
suitable for printing or attaching to a patient's file.

Uses reportlab (per the pdf skill's guidance for PDF creation).
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

INK = colors.HexColor("#0B2545")
MUTED = colors.HexColor("#5A7184")
PRIMARY = colors.HexColor("#0B5FA5")
HIGH_RED = colors.HexColor("#B23A3A")
LOW_GREEN = colors.HexColor("#2C8050")
LINE = colors.HexColor("#D6E4F0")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Times-Bold",
            fontSize=20, textColor=INK, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, textColor=MUTED, spaceAfter=16,
        ),
        "section": ParagraphStyle(
            "Section", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, textColor=PRIMARY, spaceBefore=14, spaceAfter=6,
            letterSpacing=0.5,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, textColor=INK, leading=14,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, textColor=MUTED, leading=11,
        ),
        "risk_high": ParagraphStyle(
            "RiskHigh", parent=base["Normal"], fontName="Times-Bold",
            fontSize=28, textColor=HIGH_RED,
        ),
        "risk_low": ParagraphStyle(
            "RiskLow", parent=base["Normal"], fontName="Times-Bold",
            fontSize=28, textColor=LOW_GREEN,
        ),
    }


def build_patient_report_pdf(patient: dict, prediction: dict) -> bytes:
    """Build a single-prediction PDF report.

    Args:
        patient: dict with keys name, age, gender, phone_number
        prediction: dict with keys created_at, age, glucose, bmi,
            blood_pressure, insulin, family_history, risk_label,
            probability_pct

    Returns:
        Raw PDF bytes, suitable for st.download_button.
    """
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
    )
    story = []

    # --- Header ---
    story.append(Paragraph("Diabetes Risk Screening Report", styles["title"]))
    generated_at = datetime.now().strftime("%B %d, %Y at %H:%M")
    story.append(Paragraph(f"Generated {generated_at}", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=10))

    # --- Patient details ---
    story.append(Paragraph("PATIENT DETAILS", styles["section"]))
    patient_table = Table(
        [
            ["Name", patient.get("name", "—")],
            ["Date of Birth", patient.get("date_of_birth", "—")],
            ["Age at Screening", str(prediction.get("age", "—"))],
            ["Gender", patient.get("gender", "—")],
            ["Phone Number", patient.get("phone_number", "—")],
        ],
        colWidths=[1.6 * inch, 4.2 * inch],
    )
    patient_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE),
            ]
        )
    )
    story.append(patient_table)

    # --- Result ---
    story.append(Paragraph("RISK ASSESSMENT RESULT", styles["section"]))
    risk_style = styles["risk_high"] if prediction["risk_label"] == "HIGH" else styles["risk_low"]
    result_table = Table(
        [
            [
                Paragraph("Diabetes Risk", styles["small"]),
                "",
            ],
            [
                Paragraph(prediction["risk_label"], risk_style),
                Paragraph(
                    f"Probability: {prediction['probability_pct']}%",
                    styles["body"],
                ),
            ],
        ],
        colWidths=[3 * inch, 2.8 * inch],
    )
    result_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(result_table)
    story.append(Spacer(1, 6))

    # --- Input measurements used ---
    story.append(Paragraph("HEALTH MEASUREMENTS USED", styles["section"]))
    family_history_label = "Yes" if prediction.get("family_history") else "No"
    measurements_table = Table(
        [
            ["Age", "Glucose Level", "BMI"],
            [str(prediction["age"]), f"{prediction['glucose']} mg/dL", f"{prediction['bmi']}"],
            ["Blood Pressure", "Insulin Level", "Family History"],
            [
                f"{prediction['blood_pressure']} mm Hg",
                f"{prediction['insulin']} mu U/ml",
                family_history_label,
            ],
        ],
        colWidths=[1.93 * inch, 1.93 * inch, 1.94 * inch],
    )
    measurements_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica"),
                ("FONTNAME", (0, 3), (-1, 3), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
                ("TEXTCOLOR", (0, 2), (-1, 2), MUTED),
                ("TEXTCOLOR", (0, 1), (-1, 1), INK),
                ("TEXTCOLOR", (0, 3), (-1, 3), INK),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 1), (-1, 1), 0.5, LINE),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
            ]
        )
    )
    story.append(measurements_table)

    # --- Footer / disclaimer ---
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=LINE, spaceAfter=8))
    story.append(
        Paragraph(
            "This report presents a statistical risk estimate generated by a "
            "machine learning model (TabNet) trained on historical screening "
            "data. It is not a medical diagnosis. Please consult a qualified "
            "healthcare professional for clinical evaluation and before making "
            "any medical decisions.",
            styles["small"],
        )
    )
    prediction_id = prediction.get("prediction_id", "—")
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(f"Record ID: {prediction_id}", styles["small"])
    )

    doc.build(story)
    return buf.getvalue()
