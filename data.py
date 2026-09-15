"""
data.py
-------
Handles dataset acquisition and preprocessing for the diabetes risk
prediction system, as described in Chapter 3 (Methodology) of the
accompanying research:

  Phase One   - Data Acquisition      (PIMA Indians Diabetes Dataset)
  Phase Two   - Data Preprocessing    (cleaning, missing-value handling,
                                        scaling, class-imbalance correction)

The PIMA dataset stores missing values as 0 in columns where a
physiological zero is impossible (Glucose, BloodPressure, SkinThickness,
Insulin, BMI). These are treated as missing and imputed with the median,
per the "Handling Missing Values" section (3.6.2) of the methodology.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# Columns where a recorded 0 is physiologically implausible and therefore
# treated as a missing value (Section 3.6.1 / 3.6.2 of the methodology).
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# The 6 MVP input features the app actually collects from the user.
# (SkinThickness, Pregnancies and DiabetesPedigreeFunction exist in the
# raw PIMA dataset but are not part of the MVP input form, so they are
# imputed with dataset medians at inference time rather than asked of
# the user.)
MVP_FEATURES = ["Age", "Glucose", "BMI", "BloodPressure", "Insulin", "FamilyHistory"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_PATH = os.path.join(DATA_DIR, "pima_raw.csv")

_mvp_dataset_cache = None


def download_dataset(force: bool = False) -> str:
    """Download the PIMA Indians Diabetes Dataset to data/pima_raw.csv.

    Returns the local file path. If the file already exists and
    force=False, the existing copy is reused (no network call).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_PATH) and not force:
        return RAW_PATH

    import urllib.request

    print(f"Downloading PIMA Indians Diabetes Dataset from {PIMA_URL} ...")
    try:
        urllib.request.urlretrieve(PIMA_URL, RAW_PATH)
    except Exception as e:
        raise RuntimeError(
            "Could not download the PIMA dataset automatically. "
            "If you are offline or GitHub is blocked on your network, "
            "manually download 'pima-indians-diabetes.data.csv' from "
            "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database "
            f"and save it as: {RAW_PATH}\n\nOriginal error: {e}"
        )
    print(f"Saved dataset to {RAW_PATH}")
    return RAW_PATH


def load_raw_dataframe() -> pd.DataFrame:
    """Load the raw PIMA dataset into a labeled DataFrame."""
    path = download_dataset()
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Section 3.6 preprocessing steps and engineer the
    FamilyHistory feature used by the MVP input form.

    The PIMA dataset does not contain a direct "family history yes/no"
    field; it instead has a continuous DiabetesPedigreeFunction (DPF),
    which is the standard proxy for genetic predisposition used in the
    literature reviewed in Chapter 2 (Section 2.31-iii). We binarize DPF
    at its median to create a clinically interpretable Yes/No
    FamilyHistory feature matching the MVP input form.
    """
    df = df.copy()

    # --- 3.6.1 Data Cleaning / 3.6.2 Missing Value Treatment ---
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    # --- Feature engineering: FamilyHistory from DiabetesPedigreeFunction ---
    dpf_median = df["DiabetesPedigreeFunction"].median()
    df["FamilyHistory"] = (df["DiabetesPedigreeFunction"] >= dpf_median).astype(int)

    return df


def get_mvp_dataset():
    """Return (X, y) restricted to the 6 MVP input features plus the
    Outcome label, after cleaning. This is what the TabNet model is
    actually trained on, so training and the live app use an identical
    feature set.
    """
    global _mvp_dataset_cache
    if _mvp_dataset_cache is not None:
        return _mvp_dataset_cache
    df = clean_and_engineer(load_raw_dataframe())
    X = df[MVP_FEATURES].copy()
    y = df["Outcome"].copy()
    _mvp_dataset_cache = (X, y)
    return X, y


def get_feature_medians() -> dict:
    """Median values for the MVP features, used to sanity-check /
    pre-fill the Streamlit input form.
    """
    X, _ = get_mvp_dataset()
    return X.median().to_dict()


def train_test_splits(test_size: float = 0.2, val_size: float = 0.1, seed: int = 42):
    """Stratified train/val/test split (Section 3.12 references
    stratified k-fold; for the single train/val/test split used to fit
    the deployed model we also stratify on Outcome to preserve class
    balance in every split).
    """
    X, y = get_mvp_dataset()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=seed
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler per Section 3.6.3 (Z = (X - mean) / std)."""
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    return scaler


if __name__ == "__main__":
    X, y = get_mvp_dataset()
    print("MVP feature matrix shape:", X.shape)
    print("Class balance:\n", y.value_counts(normalize=True))
    print(X.describe())
