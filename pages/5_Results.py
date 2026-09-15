"""
pages/5_Results.py
--------------------
Results page: confusion matrix, evaluation metrics, hyperparameter
tuning summary (Stratified K-Fold CV + Optuna/Grid Search), and the
ROC curve for the trained TabNet model.

Reads model/metrics.json, written by train_model.py.
"""

import json
import os

import pandas as pd
import streamlit as st

from style import metric_card

METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "metrics.json")

st.title("Model Results")
st.markdown(
    '<p class="subtitle">Held-out test performance, confusion matrix, and the '
    "hyperparameter search behind the trained model.</p>",
    unsafe_allow_html=True,
)

if not os.path.exists(METRICS_PATH):
    st.warning(
        "No metrics found yet. Run `python train_model.py` to train the model "
        "and generate this report."
    )
    st.stop()

with open(METRICS_PATH) as f:
    metrics = json.load(f)

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
st.subheader("Held-Out Test Set Metrics")
st.caption(f"Evaluated on {metrics.get('n_test', '—')} held-out patients, never seen during training or tuning.")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(metric_card("Accuracy", f"{metrics['accuracy']*100:.1f}%"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card("Precision", f"{metrics['precision']*100:.1f}%"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card("Recall", f"{metrics['recall']*100:.1f}%"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card("F1 Score", f"{metrics['f1_score']*100:.1f}%"), unsafe_allow_html=True)
with col5:
    st.markdown(metric_card("AUC-ROC", f"{metrics['auc_roc']:.3f}"), unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------
st.subheader("Confusion Matrix")

cm = metrics["confusion_matrix"]
labels = metrics.get("confusion_matrix_labels", ["LOW (0)", "HIGH (1)"])

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {l}" for l in labels],
    columns=[f"Predicted: {l}" for l in labels],
)

col_cm, col_breakdown = st.columns([3, 2])

with col_cm:
    st.dataframe(
        cm_df.style.background_gradient(cmap="Blues").format("{:d}"),
        use_container_width=True,
    )

with col_breakdown:
    tn, fp, fn, tp = metrics.get("tn"), metrics.get("fp"), metrics.get("fn"), metrics.get("tp")
    st.markdown(
        f"""
        - **True Positives (correctly flagged HIGH):** {tp}
        - **True Negatives (correctly flagged LOW):** {tn}
        - **False Positives (LOW patients flagged HIGH):** {fp}
        - **False Negatives (HIGH patients flagged LOW):** {fn}
        """
    )
    if fn is not None and (tp + fn) > 0:
        st.caption(
            f"Missed {fn} of {tp + fn} actual high-risk cases "
            f"({fn / (tp + fn) * 100:.1f}% false-negative rate)."
        )

st.write("")

# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------
if "roc_curve" in metrics:
    st.subheader("ROC Curve")
    roc = metrics["roc_curve"]
    roc_df = pd.DataFrame({"False Positive Rate": roc["fpr"], "True Positive Rate": roc["tpr"]})
    roc_df = roc_df.groupby("False Positive Rate", as_index=False)["True Positive Rate"].max()
    roc_df = roc_df.set_index("False Positive Rate")
    st.line_chart(roc_df, color="#0B5FA5")
    st.caption(f"Area under the curve (AUC-ROC): {metrics['auc_roc']:.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Hyperparameter Tuning Summary
# ---------------------------------------------------------------------------
st.subheader("Hyperparameter Tuning")

tuning = metrics.get("tuning", {})
cv_folds = metrics.get("cv_folds")
method = tuning.get("method", "unknown")

if method == "skipped":
    st.info("Tuning was skipped for this training run; default hyperparameters were used.")
else:
    method_label = "Optuna (TPE Bayesian search)" if method == "optuna" else "Grid Search"
    st.markdown(
        f"**Method:** {method_label}  ·  "
        f"**Trials evaluated:** {tuning.get('n_trials', '—')}  ·  "
        f"**Cross-validation:** Stratified {cv_folds}-Fold"
    )
    if tuning.get("best_cv_auc") is not None:
        st.markdown(f"**Best mean CV AUC-ROC:** {tuning['best_cv_auc']:.4f}")

st.markdown("**Selected hyperparameters:**")
best_params = tuning.get("best_params", {})
if best_params:
    st.dataframe(
        pd.DataFrame(
            [{"Hyperparameter": k, "Value": v} for k, v in best_params.items()]
        ),
        use_container_width=True,
        hide_index=True,
    )

all_trials = tuning.get("all_trials", [])
if all_trials:
    with st.expander(f"View all {len(all_trials)} search trials"):
        trial_rows = []
        for t in all_trials:
            row = {"Trial": t["number"], "CV AUC-ROC": round(t["value"], 4)}
            row.update(t["params"])
            trial_rows.append(row)
        trial_df = pd.DataFrame(trial_rows).sort_values("CV AUC-ROC", ascending=False)
        st.dataframe(trial_df, use_container_width=True, hide_index=True)

st.write("")
st.caption(
    "Metrics and the confusion matrix are computed once during training "
    "(`python train_model.py`) on a held-out test set that the tuning process "
    "never sees, to give an honest estimate of real-world performance."
)
