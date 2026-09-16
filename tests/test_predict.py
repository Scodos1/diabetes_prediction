"""
test_predict.py
---------------
Unit tests for prediction module (predict.py) with a mock model.
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import joblib
import pytest


class MockClassifier:
    """Mock classifier that supports predict_proba and explain."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def predict_proba(self, X):
        probs = np.random.dirichlet([1, 1], size=X.shape[0])
        return probs

    def explain(self, X):
        n_features = X.shape[1]
        explain_matrix = np.random.rand(1, n_features)
        masks = np.random.rand(X.shape[0], n_features)
        return explain_matrix, masks


@pytest.fixture
def mock_model_dir(tmp_path):
    """Create a fake model directory with mock artifacts."""
    from sklearn.preprocessing import StandardScaler

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    scaler = StandardScaler()
    X_dummy = np.random.randn(100, 6)
    scaler.fit(X_dummy)
    joblib.dump(scaler, model_dir / "scaler.joblib")

    return model_dir, scaler


@pytest.fixture
def predictor_with_mock(mock_model_dir, monkeypatch):
    """Create a DiabetesRiskPredictor with a mocked model."""
    model_dir, scaler = mock_model_dir

    import predict
    monkeypatch.setattr(predict, "MODEL_PATH", str(model_dir / "mock_model.zip"))
    monkeypatch.setattr(predict, "SCALER_PATH", str(model_dir / "scaler.joblib"))

    predictor = predict.DiabetesRiskPredictor.__new__(predict.DiabetesRiskPredictor)
    predictor.scaler = scaler
    predictor.model = MockClassifier()
    return predictor


class TestPredictRisk:
    def test_returns_dict_with_required_keys(self, predictor_with_mock):
        result = predictor_with_mock.predict_risk(
            age=40, glucose=130, bmi=28, blood_pressure=85, insulin=90, family_history=True
        )
        assert "label" in result
        assert "probability" in result
        assert "probability_pct" in result
        assert "raw_high_risk_probability" in result

    def test_label_is_high_or_low(self, predictor_with_mock):
        result = predictor_with_mock.predict_risk(
            age=40, glucose=130, bmi=28, blood_pressure=85, insulin=90, family_history=True
        )
        assert result["label"] in ("HIGH", "LOW")

    def test_probability_is_consistent(self, predictor_with_mock):
        result = predictor_with_mock.predict_risk(
            age=40, glucose=130, bmi=28, blood_pressure=85, insulin=90, family_history=True
        )
        assert result["probability"] == result["raw_high_risk_probability"]
        assert result["probability_pct"] == round(result["probability"] * 100)

    def test_probability_in_valid_range(self, predictor_with_mock):
        result = predictor_with_mock.predict_risk(
            age=40, glucose=130, bmi=28, blood_pressure=85, insulin=90, family_history=True
        )
        assert 0.0 <= result["probability"] <= 1.0
        assert 0 <= result["probability_pct"] <= 100

    def test_low_risk_input(self, predictor_with_mock):
        result = predictor_with_mock.predict_risk(
            age=25, glucose=80, bmi=22, blood_pressure=70, insulin=30, family_history=False
        )
        assert result["probability"] < 1.0


class TestExplain:
    def test_returns_feature_weights(self, predictor_with_mock):
        from data import MVP_FEATURES
        explanation = predictor_with_mock.explain(
            age=40, glucose=130, bmi=28, blood_pressure=85, insulin=90, family_history=True
        )
        assert set(explanation.keys()) == set(MVP_FEATURES)

    def test_weights_sum_to_one(self, predictor_with_mock):
        explanation = predictor_with_mock.explain(
            age=40, glucose=130, bmi=28, blood_pressure=85, insulin=90, family_history=True
        )
        total = sum(explanation.values())
        assert abs(total - 1.0) < 0.01


class TestMissingModel:
    def test_raises_if_no_model(self, tmp_path, monkeypatch):
        import predict
        monkeypatch.setattr(predict, "MODEL_PATH", str(tmp_path / "nonexistent.zip"))
        monkeypatch.setattr(predict, "SCALER_PATH", str(tmp_path / "nonexistent.joblib"))
        with pytest.raises(FileNotFoundError):
            predict.DiabetesRiskPredictor()
