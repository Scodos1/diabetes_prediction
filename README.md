# Diabetes Risk Predictor (TabNet + Streamlit)

A diabetes risk screening application built on a TabNet deep learning
model, trained on the PIMA Indians Diabetes Dataset, with patient
records, history, analytics, and PDF reporting.

## Modules

1. **Patient Registration** — Name, Date of Birth, Gender, Phone Number (age is calculated automatically wherever needed)
2. **Prediction** — select a registered patient, enter health data, run the TabNet model, see `Diabetes Risk: HIGH/LOW` + `Probability: NN%`
3. **History** — every previous prediction, searchable by patient name or phone number
4. **Analytics Dashboard** — total patients screened, high-risk count, low-risk count, plus charts (risk distribution, predictions over time, age distribution by risk, average glucose by risk)
5. **Report Generation** — download a one-page PDF report for any prediction (from the Prediction screen right after a run, or from History for any past record)
6. **Model Results** — confusion matrix, accuracy/precision/recall/F1/AUC-ROC on a held-out test set, ROC curve, and the hyperparameter search (Stratified K-Fold CV + Optuna/Grid Search) that produced the trained model

Core MVP flow: Open App → Enter Health Data → Click Predict → AI Analyzes Data → Show Risk Result

## Project structure

```
diabetes_app/
├── app.py                          # Entry point: navigation + DB init
├── pages/
│   ├── 0_Home.py                   # Landing page with quick stats
│   ├── 1_Patient_Registration.py   # Module 1
│   ├── 2_Prediction.py             # Module 2 (+ Module 5 download)
│   ├── 3_History.py                # Module 3 (+ Module 5 download)
│   ├── 4_Analytics_Dashboard.py    # Module 4
│   └── 5_Results.py                # Module 6: confusion matrix + metrics
├── data.py            # Dataset download + cleaning + preprocessing
├── tune.py             # Stratified K-Fold CV + Optuna/Grid Search tuning
├── train_model.py     # Trains and saves the TabNet model
├── predict.py          # Loads the trained model for inference
├── db.py               # SQLite persistence: patients + predictions
├── report.py           # PDF report generation (reportlab)
├── style.py            # Shared visual styling across pages (light + dark mode)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Train the model (run once)

```bash
python train_model.py                  # Optuna search, 15 trials (default)
python train_model.py --trials 30       # more Optuna trials = better search
python train_model.py --grid-search     # force Grid Search instead of Optuna
python train_model.py --skip-tuning     # skip tuning, use fixed default hyperparameters
```

This will:
- Download the PIMA Indians Diabetes Dataset (768 records) to `data/pima_raw.csv`
- Clean it (treat implausible 0s in Glucose/BloodPressure/SkinThickness/Insulin/BMI as missing, impute with median)
- Engineer the `FamilyHistory` Yes/No feature from the dataset's DiabetesPedigreeFunction
- **Tune hyperparameters** using Optuna's Bayesian (TPE) search if Optuna is
  installed, or an exhaustive Grid Search otherwise — every candidate is
  scored with **Stratified 5-Fold Cross Validation** (mean AUC-ROC across
  folds), so tuning decisions aren't based on a single lucky split
- Balance classes with SMOTE (re-applied per fold during tuning, and once
  more on the final train+val set, to avoid leaking resampled data into
  validation folds)
- Train a final TabNet classifier using the best hyperparameters found
- Save the trained model + scaler to `model/`
- Evaluate once on a held-out test set the tuning process never touches,
  and save accuracy, precision, recall, F1, AUC-ROC, the confusion matrix,
  and the full tuning history to `model/metrics.json`
- Open the app's **Model Results** page to view all of this

**If your network blocks GitHub:** download
`pima-indians-diabetes.data.csv` manually from
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
and place it at `data/pima_raw.csv` before running `train_model.py`.

**If Optuna isn't installed:** `train_model.py` automatically falls back to
Grid Search over a fixed set of candidates in `tune.py`'s `PARAM_GRID` — no
separate flag needed, though `--grid-search` forces this path explicitly.

## Run the app

```bash
streamlit run app.py
```

This opens the app in your browser, typically at `http://localhost:8501`.
Use the sidebar to move between modules. The database (`app_data.db`,
created automatically) stores all patients and predictions locally as
a single SQLite file in the project folder.

## Notes on the model

- **Why TabNet:** TabNet uses sequential attention to select which
  features to reason about at each decision step, which gives
  per-prediction feature importance (shown in the Prediction page's
  "What influenced this prediction?" panel) without needing a separate
  explainability tool like SHAP.
- **Why these 6 inputs:** The MVP form intentionally exposes a small,
  easy-to-self-report subset of the full PIMA feature set. The model
  is trained on exactly these 6 features so there's no mismatch
  between training and inference.
- **FamilyHistory feature:** PIMA doesn't have a literal yes/no family
  history field. It has a continuous Diabetes Pedigree Function (DPF),
  the standard genetic-risk proxy used in PIMA-based research. The
  training pipeline binarizes DPF at its median to produce a
  Yes/No FamilyHistory label.
- **Hyperparameter tuning:** `tune.py` scores every candidate hyperparameter
  set with Stratified 5-Fold Cross Validation (mean AUC-ROC), then either
  Optuna (Bayesian TPE search, if installed) or Grid Search (fixed
  candidate list, always available) picks the best one. The held-out test
  set used for final evaluation is never part of this search, so the
  reported metrics reflect genuinely unseen data.
- **Dark mode:** the app's theme follows the system/browser color scheme.
  All custom colors in `style.py` have explicit light- and dark-mode
  variants tuned to maintain WCAG AA text contrast in both.

## Data storage

`app_data.db` (SQLite) holds two tables:
- `patients` — one row per registered patient
- `predictions` — one row per prediction run, linked to a patient

This file is created automatically on first run and persists between
sessions. Back it up like any other file if you want to preserve
patient history.

## Disclaimer

This tool produces a statistical risk estimate from a machine learning
model. It is **not** a medical diagnosis and should not replace
consultation with a qualified healthcare professional.
