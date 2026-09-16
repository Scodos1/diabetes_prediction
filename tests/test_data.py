"""
test_data.py
------------
Unit tests for data preprocessing (data.py).
"""

import os
import numpy as np
import pandas as pd
import pytest


class TestCleanAndEngineer:
    def test_replaces_zero_with_nan(self, sample_df):
        from data import clean_and_engineer
        df = sample_df.copy()
        df.loc[0, "Glucose"] = 0
        df.loc[1, "BloodPressure"] = 0
        df.loc[2, "Insulin"] = 0
        result = clean_and_engineer(df)
        assert not (result["Glucose"] == 0).any()
        assert not (result["BloodPressure"] == 0).any()
        assert not (result["Insulin"] == 0).any()

    def test_adds_family_history_column(self, sample_df):
        from data import clean_and_engineer
        result = clean_and_engineer(sample_df)
        assert "FamilyHistory" in result.columns
        assert set(result["FamilyHistory"].unique()).issubset({0, 1})

    def test_family_history_binary_at_median(self, sample_df):
        from data import clean_and_engineer
        result = clean_and_engineer(sample_df)
        dpf_median = sample_df["DiabetesPedigreeFunction"].median()
        expected = (sample_df["DiabetesPedigreeFunction"] >= dpf_median).astype(int)
        pd.testing.assert_series_equal(
            result["FamilyHistory"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_no_nans_in_cleaned_columns(self, sample_df):
        from data import clean_and_engineer, ZERO_AS_MISSING
        result = clean_and_engineer(sample_df)
        for col in ZERO_AS_MISSING:
            assert not result[col].isna().any(), f"NaN found in {col}"


class TestGetMvpDataset:
    def test_returns_tuple(self):
        from data import get_mvp_dataset
        X, y = get_mvp_dataset()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_correct_features(self):
        from data import get_mvp_dataset, MVP_FEATURES
        X, y = get_mvp_dataset()
        assert list(X.columns) == MVP_FEATURES

    def test_no_nans(self):
        from data import get_mvp_dataset
        X, y = get_mvp_dataset()
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_y_is_binary(self):
        from data import get_mvp_dataset
        X, y = get_mvp_dataset()
        assert set(y.unique()).issubset({0, 1})

    def test_caching(self):
        from data import get_mvp_dataset
        X1, y1 = get_mvp_dataset()
        X2, y2 = get_mvp_dataset()
        assert X1 is X2
        assert y1 is y2


class TestTrainTestSplits:
    def test_returns_six_items(self):
        from data import train_test_splits
        result = train_test_splits()
        assert len(result) == 6

    def test_correct_shapes(self):
        from data import train_test_splits
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_splits()
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 768

    def test_stratified(self):
        from data import train_test_splits
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_splits()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.1


class TestFitScaler:
    def test_returns_scaler(self):
        from data import train_test_splits, fit_scaler
        X_train, _, _, _, _, _ = train_test_splits()
        scaler = fit_scaler(X_train)
        assert hasattr(scaler, "transform")
        assert hasattr(scaler, "mean_")

    def test_output_is_standardized(self):
        from data import train_test_splits, fit_scaler
        X_train, _, _, _, _, _ = train_test_splits()
        scaler = fit_scaler(X_train)
        X_scaled = scaler.transform(X_train.values)
        assert abs(X_scaled.mean()) < 0.1
        assert abs(X_scaled.std() - 1.0) < 0.2
