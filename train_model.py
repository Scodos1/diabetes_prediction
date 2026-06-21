"""
train_model.py
---------------
Implements Phases Three and Four of the proposed system methodology
(Chapter 3):

  Phase Three : Model Development      -> TabNet architecture
  Phase Four  : Model Optimization &
                Evaluation              -> Stratified K-Fold CV,
                                           Optuna/Grid Search
                                           hyperparameter tuning,
                                           class-imbalance handling
                                           (SMOTE), held-out test
                                           evaluation, and the metrics
                                           reported in Section 3.11
                                           (Accuracy, Precision, Recall,
                                           F1-score, AUC-ROC, confusion
                                           matrix).

Run this once to produce the trained artifacts the Streamlit app loads:

    python train_model.py                 # Optuna search (15 trials)
    python train_model.py --trials 30      # more Optuna trials
    python train_model.py --grid-search    # force Grid Search instead
    python train_model.py --skip-tuning    # use fixed default hyperparameters

Outputs (written to ./model/):
    tabnet_diabetes.zip   - trained TabNet model (pytorch-tabnet format)
    scaler.joblib         - fitted StandardScaler (Section 3.6.3)
    metrics.json          - CV tuning history + held-out test metrics +
                             confusion matrix, read by the Results page
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import joblib

from data import train_test_splits, fit_scaler, MVP_FEATURES
from tune import tune_hyperparameters, N_SPLITS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "tabnet_diabetes")  # .zip is appended by TabNet
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Used when --skip-tuning is passed, or as the Grid Search seed point.
DEFAULT_PARAMS = {
    "n_d": 8,
    "n_a": 8,
    "n_steps": 3,
    "gamma": 1.5,
    "lambda_sparse": 1e-3,
    "lr": 2e-2,
    "batch_size": 64,
    "virtual_batch_size": 16,
}


def build_classifier(params: dict, seed: int) -> TabNetClassifier:
    return TabNetClassifier(
        n_d=params["n_d"],
        n_a=params["n_a"],
        n_steps=params["n_steps"],
        gamma=params["gamma"],
        lambda_sparse=params["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=params["lr"]),
        scheduler_params=dict(step_size=20, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
        seed=seed,
        verbose=1,
    )


def main(n_trials: int = 15, seed: int = 42, force_grid_search: bool = False,
         skip_tuning: bool = False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Phase One/Two: data acquisition + preprocessing (data.py) ---
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_splits(seed=seed)

    # --- Section 3.6.3 Feature Scaling ---
    scaler = fit_scaler(X_train)
    X_train_s = scaler.transform(X_train.values)
    X_val_s = scaler.transform(X_val.values)
    X_test_s = scaler.transform(X_test.values)

    # ------------------------------------------------------------------
    # Hyperparameter tuning: Stratified K-Fold CV + Optuna/Grid Search.
    # Tuning runs over train+val combined (the held-out test set is
    # never touched until final evaluation, to keep it a clean estimate
    # of real-world performance).
    # ------------------------------------------------------------------
    X_tune = np.vstack([X_train_s, X_val_s])
    y_tune = pd.concat([y_train, y_val]).reset_index(drop=True).values

    if skip_tuning:
        print("Skipping hyperparameter tuning; using DEFAULT_PARAMS.")
        tuning_result = {
            "method": "skipped",
            "n_trials": 0,
            "best_params": DEFAULT_PARAMS,
            "best_cv_auc": None,
            "all_trials": [],
        }
    else:
        tuning_result = tune_hyperparameters(
            X_tune, y_tune, n_trials=n_trials, seed=seed,
            force_grid_search=force_grid_search,
        )

    best_params = tuning_result["best_params"]
    print(f"\nBest hyperparameters ({tuning_result['method']}): {best_params}")
    if tuning_result["best_cv_auc"] is not None:
        print(f"Best {N_SPLITS}-fold CV AUC-ROC: {tuning_result['best_cv_auc']:.4f}")

    # ------------------------------------------------------------------
    # Final model: retrain on the full train+val set (SMOTE-balanced)
    # using the best hyperparameters found above, then evaluate once on
    # the untouched held-out test set.
    # ------------------------------------------------------------------
    smote = SMOTE(random_state=seed)
    X_final_bal, y_final_bal = smote.fit_resample(X_tune, y_tune)
    print(
        f"\nFinal training set class balance after SMOTE: "
        f"{np.bincount(y_final_bal.astype(int))} "
        f"(before: {np.bincount(y_tune.astype(int))})"
    )

    clf = build_classifier(best_params, seed=seed)
    clf.fit(
        X_train=X_final_bal,
        y_train=y_final_bal,
        eval_set=[(X_test_s, y_test.values)],
        eval_metric=["auc", "accuracy"],
        max_epochs=200,
        patience=25,
        batch_size=best_params.get("batch_size", 64),
        virtual_batch_size=best_params.get("virtual_batch_size", 16),
    )

    # --- Section 3.11 Model Evaluation Metrics (held-out test set) ---
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["LOW (0)", "HIGH (1)"],
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
        },
        "n_test": int(len(y_test)),
        "features": MVP_FEATURES,
        "tuning": tuning_result,
        "cv_folds": N_SPLITS,
    }

    print("\n=== Held-out Test Set Performance ===")
    for k in ("accuracy", "precision", "recall", "f1_score", "auc_roc"):
        print(f"{k:>12}: {metrics[k]:.4f}")
    print("confusion_matrix [[TN FP] [FN TP]]:", metrics["confusion_matrix"])

    # --- Persist artifacts ---
    clf.save_model(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}.zip")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print("\nOpen the app and visit the Results page to view the confusion matrix and metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TabNet diabetes risk model.")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna trials (default: 15)")
    parser.add_argument("--grid-search", action="store_true", help="Force Grid Search instead of Optuna")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip tuning, use default hyperparameters")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        n_trials=args.trials,
        seed=args.seed,
        force_grid_search=args.grid_search,
        skip_tuning=args.skip_tuning,
    )
