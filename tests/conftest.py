"""
conftest.py
-----------
Shared pytest fixtures for the diabetes app test suite.
"""

import os
import sqlite3
import tempfile

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def tmp_db(tmp_path):
    """Create a fresh temporary SQLite database for each test."""
    db_path = str(tmp_path / "test_app.db")

    import db
    original_db_path = db.DB_PATH
    db.DB_PATH = db_path
    db.clear_cache()
    db.init_db()

    yield db_path

    db.DB_PATH = original_db_path
    db.clear_cache()


@pytest.fixture
def sample_patient():
    """A valid patient dict for registration."""
    return {
        "name": "Jane Doe",
        "date_of_birth": "1985-03-15",
        "gender": "Female",
        "phone_number": "555-1234",
    }


@pytest.fixture
def sample_prediction_inputs():
    """Valid model inputs for a prediction."""
    return {
        "age": 38,
        "glucose": 130.0,
        "bmi": 28.5,
        "blood_pressure": 85.0,
        "insulin": 90.0,
        "family_history": True,
    }


@pytest.fixture
def sample_prediction_result():
    """A model result dict."""
    return {
        "label": "HIGH",
        "probability": 0.78,
        "probability_pct": 78,
        "raw_high_risk_probability": 0.78,
    }


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the write rate limiter between tests."""
    import db
    db._write_timestamps.clear()
    yield
    db._write_timestamps.clear()


@pytest.fixture
def sample_df():
    """A small synthetic DataFrame mimicking cleaned PIMA data."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "Pregnancies": np.random.randint(0, 10, n),
        "Glucose": np.random.randint(70, 200, n).astype(float),
        "BloodPressure": np.random.randint(50, 120, n).astype(float),
        "SkinThickness": np.random.randint(10, 50, n).astype(float),
        "Insulin": np.random.randint(20, 300, n).astype(float),
        "BMI": np.random.uniform(18, 45, n),
        "DiabetesPedigreeFunction": np.random.uniform(0.1, 2.5, n),
        "Age": np.random.randint(20, 70, n),
        "Outcome": np.random.randint(0, 2, n),
    })
