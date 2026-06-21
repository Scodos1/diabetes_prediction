"""
db.py
-----
SQLite persistence layer for the clinical modules:

  Module 1 - Patient Registration   (patients table)
  Module 3 - History Module          (predictions table + search)
  Module 4 - Analytics Dashboard     (aggregate queries over predictions)

A single local file, app_data.db, stores everything. SQLite is used
rather than an external DB server since this is a single-clinician /
single-machine MVP deployment (matching the Streamlit single-process
model); swapping in Postgres later only requires changing the
connection logic in `get_connection()`.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "app_data.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    date_of_birth   TEXT NOT NULL,
    gender          TEXT NOT NULL,
    phone_number    TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id              INTEGER NOT NULL,
    created_at               TEXT NOT NULL,
    age                      REAL NOT NULL,
    glucose                  REAL NOT NULL,
    bmi                      REAL NOT NULL,
    blood_pressure           REAL NOT NULL,
    insulin                  REAL NOT NULL,
    family_history           INTEGER NOT NULL,
    risk_label                TEXT NOT NULL,
    probability_pct           INTEGER NOT NULL,
    raw_high_risk_probability REAL NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions (patient_id);
CREATE INDEX IF NOT EXISTS idx_patients_name ON patients (name);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_age(date_of_birth: str, as_of: "date | None" = None) -> int:
    """Compute whole years of age from an ISO 'YYYY-MM-DD' date_of_birth
    string. Used to derive the Age feature the TabNet model expects
    from the Date of Birth captured at registration.
    """
    from datetime import date as _date

    if isinstance(date_of_birth, str):
        dob = _date.fromisoformat(date_of_birth)
    else:
        dob = date_of_birth
    today = as_of or _date.today()
    years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return years


# ---------------------------------------------------------------------------
# Module 1: Patient Registration
# ---------------------------------------------------------------------------

def register_patient(name: str, date_of_birth: str, gender: str, phone_number: str) -> int:
    """Insert a new patient record. date_of_birth must be an ISO
    'YYYY-MM-DD' string. Returns the new patient_id.
    """
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO patients (name, date_of_birth, gender, phone_number, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name.strip(), date_of_birth, gender, phone_number.strip(), _now()),
        )
        return cur.lastrowid


def list_patients(search_text: str = "") -> list:
    """Return all patients, optionally filtered by name/phone substring
    (case-insensitive), most recently registered first.
    """
    with get_connection() as conn:
        if search_text:
            rows = conn.execute(
                "SELECT * FROM patients "
                "WHERE name LIKE ? OR phone_number LIKE ? "
                "ORDER BY created_at DESC",
                (f"%{search_text}%", f"%{search_text}%"),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM patients ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def get_patient(patient_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Module 2 / 3: Prediction storage + History
# ---------------------------------------------------------------------------

def save_prediction(patient_id: int, inputs: dict, result: dict) -> int:
    """Persist one prediction run (Module 2 output) so it can appear in
    the History Module (Module 3) and Analytics Dashboard (Module 4).
    """
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                patient_id, created_at, age, glucose, bmi, blood_pressure,
                insulin, family_history, risk_label, probability_pct,
                raw_high_risk_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                _now(),
                inputs["age"],
                inputs["glucose"],
                inputs["bmi"],
                inputs["blood_pressure"],
                inputs["insulin"],
                1 if inputs["family_history"] else 0,
                result["label"],
                result["probability_pct"],
                result["raw_high_risk_probability"],
            ),
        )
        return cur.lastrowid


def get_predictions_for_patient(patient_id: int) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE patient_id = ? ORDER BY created_at DESC",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_prediction(prediction_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,)
        ).fetchone()
        return dict(row) if row else None


def search_history(search_text: str = "") -> list:
    """Module 3: search patient records by name/phone, joined with
    their prediction history. Returns one row per prediction, newest
    first, with patient details attached.
    """
    with get_connection() as conn:
        query = """
            SELECT p.prediction_id, p.created_at, p.age, p.glucose, p.bmi,
                   p.blood_pressure, p.insulin, p.family_history,
                   p.risk_label, p.probability_pct,
                   pa.patient_id, pa.name, pa.gender, pa.phone_number
            FROM predictions p
            JOIN patients pa ON pa.patient_id = p.patient_id
        """
        params = ()
        if search_text:
            query += " WHERE pa.name LIKE ? OR pa.phone_number LIKE ?"
            params = (f"%{search_text}%", f"%{search_text}%")
        query += " ORDER BY p.created_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Module 4: Analytics Dashboard
# ---------------------------------------------------------------------------

def get_analytics_summary() -> dict:
    """Aggregate counts for the dashboard: total patients screened
    (distinct patients with at least one prediction), high-risk count,
    low-risk count, and total prediction runs.
    """
    with get_connection() as conn:
        total_patients = conn.execute(
            "SELECT COUNT(DISTINCT patient_id) AS c FROM predictions"
        ).fetchone()["c"]
        high_risk = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE risk_label = 'HIGH'"
        ).fetchone()["c"]
        low_risk = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE risk_label = 'LOW'"
        ).fetchone()["c"]
        total_predictions = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions"
        ).fetchone()["c"]
        return {
            "total_patients_screened": total_patients,
            "high_risk_count": high_risk,
            "low_risk_count": low_risk,
            "total_predictions": total_predictions,
        }


def get_predictions_over_time() -> list:
    """Predictions grouped by calendar day, for a trend chart."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day,
                   SUM(CASE WHEN risk_label = 'HIGH' THEN 1 ELSE 0 END) AS high_count,
                   SUM(CASE WHEN risk_label = 'LOW' THEN 1 ELSE 0 END) AS low_count
            FROM predictions
            GROUP BY day
            ORDER BY day ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_predictions() -> list:
    """All predictions joined with patient info, for analytics charts
    that need per-record detail (e.g. age distribution by risk).
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.*, pa.name, pa.gender
            FROM predictions p
            JOIN patients pa ON pa.patient_id = p.patient_id
            ORDER BY p.created_at ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]
