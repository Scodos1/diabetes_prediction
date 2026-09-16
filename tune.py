"""
tune.py
-------
Hyperparameter tuning for the TabNet model, using:

  1. Stratified K-Fold Cross Validation  - each candidate hyperparameter
     set is scored by average AUC-ROC across K stratified folds of the
     training data, so the search isn't fooled by a lucky single split.

  2. Optuna (preferred) or Grid Search (fallback) - searches the
     hyperparameter space defined in TABNET_SEARCH_SPACE / PARAM_GRID.

If Optuna is not installed, this module automatically falls back to an
exhaustive Grid Search over a smaller, discretized version of the same
space, so `train_model.py` works either way.
"""

import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

log = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

N_SPLITS = 5  # Stratified K-Fold count


# ---------------------------------------------------------------------------
# Shared: train+evaluate one hyperparameter set across K stratified folds
# ---------------------------------------------------------------------------

def _cv_score(params: dict, X: np.ndarray, y: np.ndarray, seed: int = 42,
              max_epochs: int = 80, patience: int = 15) -> float:
    """Stratified K-Fold CV: returns the mean AUC-ROC of `params` across
    N_SPLITS folds. Each fold gets its own SMOTE resampling (fit only on
    that fold's training portion, never the validation portion, to avoid
    leakage) before a TabNet model is trained and scored on the held-out
    fold.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        smote = SMOTE(random_state=seed)
        X_tr_bal, y_tr_bal = smote.fit_resample(X_tr, y_tr)

        clf = TabNetClassifier(
            n_d=params["n_d"],
            n_a=params["n_a"],
            n_steps=params["n_steps"],
            gamma=params["gamma"],
            lambda_sparse=params["lambda_sparse"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=params["lr"]),
            mask_type="sparsemax",
            seed=seed,
            verbose=0,
        )
        clf.fit(
            X_train=X_tr_bal,
            y_train=y_tr_bal,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=params.get("batch_size", 64),
            virtual_batch_size=params.get("virtual_batch_size", 16),
        )
        proba = clf.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, proba)
        fold_scores.append(score)
        log.info("    fold %d/%d AUC: %.4f", fold_idx + 1, N_SPLITS, score)

    return float(np.mean(fold_scores))


# ---------------------------------------------------------------------------
# Option 1: Optuna search (Bayesian / TPE), preferred when available
# ---------------------------------------------------------------------------

def _optuna_search(X: np.ndarray, y: np.ndarray, n_trials: int, seed: int) -> dict:
    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_d": trial.suggest_categorical("n_d", [8, 16, 24]),
            "n_a": trial.suggest_categorical("n_a", [8, 16, 24]),
            "n_steps": trial.suggest_int("n_steps", 3, 6),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-4, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 5e-3, 3e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "virtual_batch_size": 16,
        }
        log.info("  [Optuna trial %d] params=%s", trial.number, params)
        return _cv_score(params, X, y, seed=seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name="tabnet_diabetes_tuning",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.setdefault("virtual_batch_size", 16)
    return {
        "method": "optuna",
        "n_trials": n_trials,
        "best_params": best_params,
        "best_cv_auc": float(study.best_value),
        "all_trials": [
            {"number": t.number, "params": t.params, "value": t.value}
            for t in study.trials
            if t.value is not None
        ],
    }


# ---------------------------------------------------------------------------
# Option 2: Grid Search fallback (no external dependency beyond sklearn)
# ---------------------------------------------------------------------------

PARAM_GRID = [
    {"n_d": 8, "n_a": 8, "n_steps": 3, "gamma": 1.5, "lambda_sparse": 1e-3,
     "lr": 2e-2, "batch_size": 64, "virtual_batch_size": 16},
    {"n_d": 16, "n_a": 16, "n_steps": 3, "gamma": 1.5, "lambda_sparse": 1e-3,
     "lr": 2e-2, "batch_size": 64, "virtual_batch_size": 16},
    {"n_d": 8, "n_a": 8, "n_steps": 5, "gamma": 1.3, "lambda_sparse": 1e-4,
     "lr": 1e-2, "batch_size": 32, "virtual_batch_size": 16},
    {"n_d": 16, "n_a": 16, "n_steps": 5, "gamma": 1.8, "lambda_sparse": 1e-3,
     "lr": 1e-2, "batch_size": 128, "virtual_batch_size": 16},
    {"n_d": 24, "n_a": 24, "n_steps": 4, "gamma": 1.5, "lambda_sparse": 1e-2,
     "lr": 5e-3, "batch_size": 64, "virtual_batch_size": 16},
    {"n_d": 8, "n_a": 8, "n_steps": 4, "gamma": 2.0, "lambda_sparse": 1e-3,
     "lr": 3e-2, "batch_size": 64, "virtual_batch_size": 16},
]


def _grid_search(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    results = []
    for i, params in enumerate(PARAM_GRID):
        log.info("  [Grid Search candidate %d/%d] params=%s", i + 1, len(PARAM_GRID), params)
        score = _cv_score(params, X, y, seed=seed)
        results.append({"params": params, "value": score})

    best = max(results, key=lambda r: r["value"])
    return {
        "method": "grid_search",
        "n_trials": len(PARAM_GRID),
        "best_params": best["params"],
        "best_cv_auc": best["value"],
        "all_trials": [
            {"number": i, "params": r["params"], "value": r["value"]}
            for i, r in enumerate(results)
        ],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def tune_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 15,
                          seed: int = 42, force_grid_search: bool = False) -> dict:
    """Run hyperparameter tuning using Stratified K-Fold CV.

    Uses Optuna's Tree-structured Parzen Estimator sampler when Optuna
    is installed; otherwise falls back to an exhaustive Grid Search
    over PARAM_GRID. Either way, every candidate is scored with
    N_SPLITS-fold Stratified K-Fold CV (mean AUC-ROC).

    Returns a dict with: method, n_trials, best_params, best_cv_auc,
    all_trials (full search history, useful for the Results page).
    """
    if OPTUNA_AVAILABLE and not force_grid_search:
        log.info("Running Optuna search (%d trials, %d-fold Stratified CV per trial)...", n_trials, N_SPLITS)
        return _optuna_search(X, y, n_trials=n_trials, seed=seed)
    else:
        if not OPTUNA_AVAILABLE:
            log.warning("Optuna not installed - falling back to Grid Search.")
        log.info("Running Grid Search (%d candidates, %d-fold Stratified CV each)...", len(PARAM_GRID), N_SPLITS)
        return _grid_search(X, y, seed=seed)
