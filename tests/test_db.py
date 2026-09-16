"""
test_db.py
----------
Unit tests for the database layer (db.py).
"""

import os
import sqlite3
from datetime import date

import pytest


class TestInitDB:
    def test_creates_tables(self, tmp_db):
        import db
        conn = sqlite3.connect(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        conn.close()
        assert "patients" in table_names
        assert "predictions" in table_names

    def test_idempotent(self, tmp_db):
        import db
        db.init_db()
        db.init_db()


class TestRegisterPatient:
    def test_returns_patient_id(self, tmp_db, sample_patient):
        import db
        pid = db.register_patient(**sample_patient)
        assert isinstance(pid, int)
        assert pid > 0

    def test_strips_name_whitespace(self, tmp_db):
        import db
        pid = db.register_patient("  Alice Smith  ", "1990-01-01", "Female", "555-0000")
        patient = db.get_patient(pid)
        assert patient["name"] == "Alice Smith"

    def test_strips_phone_whitespace(self, tmp_db):
        import db
        pid = db.register_patient("Bob", "1990-01-01", "Male", "  555-1111  ")
        patient = db.get_patient(pid)
        assert patient["phone_number"] == "555-1111"


class TestGetPatient:
    def test_existing_patient(self, tmp_db, sample_patient):
        import db
        pid = db.register_patient(**sample_patient)
        patient = db.get_patient(pid)
        assert patient is not None
        assert patient["name"] == sample_patient["name"]
        assert patient["gender"] == sample_patient["gender"]

    def test_nonexistent_patient(self, tmp_db):
        import db
        assert db.get_patient(99999) is None


class TestListPatients:
    def test_empty_database(self, tmp_db):
        import db
        assert db.list_patients() == []

    def test_lists_all_patients(self, tmp_db):
        import db
        db.register_patient("A", "1990-01-01", "Male", "111")
        db.register_patient("B", "1985-06-15", "Female", "222")
        patients = db.list_patients()
        assert len(patients) == 2

    def test_search_by_name(self, tmp_db):
        import db
        db.register_patient("Alice", "1990-01-01", "Female", "111")
        db.register_patient("Bob", "1985-06-15", "Male", "222")
        results = db.list_patients("Alice")
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

    def test_search_by_phone(self, tmp_db):
        import db
        db.register_patient("Alice", "1990-01-01", "Female", "555-1234")
        db.register_patient("Bob", "1985-06-15", "Male", "555-9999")
        results = db.list_patients("1234")
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

    def test_case_insensitive_search(self, tmp_db):
        import db
        db.register_patient("Alice", "1990-01-01", "Female", "111")
        results = db.list_patients("alice")
        assert len(results) == 1


class TestSavePrediction:
    def test_returns_prediction_id(self, tmp_db, sample_patient, sample_prediction_inputs, sample_prediction_result):
        import db
        pid = db.register_patient(**sample_patient)
        pred_id = db.save_prediction(pid, sample_prediction_inputs, sample_prediction_result)
        assert isinstance(pred_id, int)
        assert pred_id > 0

    def test_stores_correctly(self, tmp_db, sample_patient, sample_prediction_inputs, sample_prediction_result):
        import db
        pid = db.register_patient(**sample_patient)
        pred_id = db.save_prediction(pid, sample_prediction_inputs, sample_prediction_result)
        pred = db.get_prediction(pred_id)
        assert pred is not None
        assert pred["patient_id"] == pid
        assert pred["risk_label"] == "HIGH"
        assert pred["glucose"] == 130.0


class TestGetPrediction:
    def test_existing_prediction(self, tmp_db, sample_patient, sample_prediction_inputs, sample_prediction_result):
        import db
        pid = db.register_patient(**sample_patient)
        pred_id = db.save_prediction(pid, sample_prediction_inputs, sample_prediction_result)
        pred = db.get_prediction(pred_id)
        assert pred is not None
        assert pred["prediction_id"] == pred_id

    def test_nonexistent_prediction(self, tmp_db):
        import db
        assert db.get_prediction(99999) is None


class TestGetPredictionsForPatient:
    def test_returns_predictions(self, tmp_db, sample_patient, sample_prediction_inputs, sample_prediction_result):
        import db
        pid = db.register_patient(**sample_patient)
        db.save_prediction(pid, sample_prediction_inputs, sample_prediction_result)
        db.save_prediction(pid, sample_prediction_inputs, sample_prediction_result)
        preds = db.get_predictions_for_patient(pid)
        assert len(preds) == 2

    def test_empty_for_new_patient(self, tmp_db, sample_patient):
        import db
        pid = db.register_patient(**sample_patient)
        assert db.get_predictions_for_patient(pid) == []


class TestSearchHistory:
    def test_empty(self, tmp_db):
        import db
        assert db.search_history() == []

    def test_returns_joined_data(self, tmp_db, sample_patient, sample_prediction_inputs, sample_prediction_result):
        import db
        pid = db.register_patient(**sample_patient)
        db.save_prediction(pid, sample_prediction_inputs, sample_prediction_result)
        results = db.search_history()
        assert len(results) == 1
        assert results[0]["name"] == sample_patient["name"]

    def test_search_filter(self, tmp_db, sample_patient, sample_prediction_inputs, sample_prediction_result):
        import db
        pid = db.register_patient(**sample_patient)
        db.save_prediction(pid, sample_prediction_inputs, sample_prediction_result)
        db.register_patient("Other Guy", "1990-01-01", "Male", "999-9999")
        results = db.search_history("Jane")
        assert len(results) == 1


class TestAnalytics:
    def test_empty_summary(self, tmp_db):
        import db
        summary = db.get_analytics_summary()
        assert summary["total_patients_screened"] == 0
        assert summary["high_risk_count"] == 0
        assert summary["low_risk_count"] == 0
        assert summary["total_predictions"] == 0

    def test_summary_with_data(self, tmp_db, sample_patient, sample_prediction_inputs):
        import db
        pid = db.register_patient(**sample_patient)
        high_result = {"label": "HIGH", "probability_pct": 80, "raw_high_risk_probability": 0.8}
        low_result = {"label": "LOW", "probability_pct": 20, "raw_high_risk_probability": 0.2}
        db.save_prediction(pid, sample_prediction_inputs, high_result)
        db.save_prediction(pid, sample_prediction_inputs, low_result)
        summary = db.get_analytics_summary()
        assert summary["total_patients_screened"] == 1
        assert summary["high_risk_count"] == 1
        assert summary["low_risk_count"] == 1
        assert summary["total_predictions"] == 2

    def test_predictions_over_time(self, tmp_db, sample_patient, sample_prediction_inputs):
        import db
        pid = db.register_patient(**sample_patient)
        result = {"label": "HIGH", "probability_pct": 80, "raw_high_risk_probability": 0.8}
        db.save_prediction(pid, sample_prediction_inputs, result)
        trend = db.get_predictions_over_time()
        assert len(trend) == 1
        assert "day" in trend[0]
        assert "high_count" in trend[0]


class TestCalculateAge:
    def test_known_date(self):
        import db
        age = db.calculate_age("2000-01-01", as_of=date(2025, 6, 15))
        assert age == 25

    def test_birthday_not_yet(self):
        import db
        age = db.calculate_age("2000-06-15", as_of=date(2025, 6, 14))
        assert age == 24

    def test_birthday_passed(self):
        import db
        age = db.calculate_age("2000-06-15", as_of=date(2025, 6, 15))
        assert age == 25


class TestDeletePatientCascade:
    def test_delete_patient_cascades(self, tmp_db, sample_patient, sample_prediction_inputs, sample_prediction_result):
        import db
        pid = db.register_patient(**sample_patient)
        db.save_prediction(pid, sample_prediction_inputs, sample_prediction_result)
        deleted = db.delete_patient(pid)
        assert deleted is True
        assert db.get_patient(pid) is None
        assert db.get_predictions_for_patient(pid) == []

    def test_delete_nonexistent_returns_false(self, tmp_db):
        import db
        assert db.delete_patient(99999) is False


class TestInputValidation:
    def test_name_too_long_raises(self, tmp_db):
        import db
        with pytest.raises(ValueError, match="100 characters"):
            db.register_patient("A" * 101, "1990-01-01", "Male", "555-0000")

    def test_phone_too_long_raises(self, tmp_db):
        import db
        with pytest.raises(ValueError, match="20 characters"):
            db.register_patient("Bob", "1990-01-01", "Male", "1" * 21)

    def test_name_whitespace_collapsed(self, tmp_db):
        import db
        pid = db.register_patient("  Alice   Smith  ", "1990-01-01", "Female", "111")
        patient = db.get_patient(pid)
        assert patient["name"] == "Alice Smith"


class TestRateLimiting:
    def test_rate_limit_exceeded(self, tmp_db):
        import db
        db._RATE_LIMIT_MAX_WRITES = 3
        db._write_timestamps.clear()
        for _ in range(3):
            db.register_patient("Test", "1990-01-01", "Male", "111")
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            db.register_patient("Blocked", "1990-01-01", "Male", "222")
        db._RATE_LIMIT_MAX_WRITES = 30
