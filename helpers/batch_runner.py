"""Pure batch processing functions — no Streamlit code."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from scripts.feature_engineering import transform_features_for_scoring
from scripts.fraud_investigation_agents import investigate_transaction, run_agents


def run_ml_batch(
    data: dict,
    models: dict,
    n_samples: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a random sample of test transactions.

    Uses stratified sampling so that both classes are represented whenever
    the test set contains positives, preventing all-NaN metrics on small batches.

    Returns (idx, y_sample, cal_scores) — all numpy arrays of length n_samples.
    """
    rng = np.random.default_rng(seed)
    y_all = data["y_test"].values
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]

    if len(pos_idx) > 0 and n_samples >= 2:
        # Guarantee at least 1 positive; scale the rest to match the natural ratio.
        n_pos = max(1, round(n_samples * len(pos_idx) / len(y_all)))
        n_pos = min(n_pos, len(pos_idx), n_samples - 1)
        n_neg = n_samples - n_pos
        chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        chosen_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        idx = np.concatenate([chosen_pos, chosen_neg])
        rng.shuffle(idx)
    else:
        idx = rng.choice(len(y_all), size=n_samples, replace=False)

    X_all = transform_features_for_scoring(data["X_test"], models["feature_names"])
    X_sample = X_all.iloc[idx]
    y_sample = data["y_test"].iloc[idx].values

    raw_scores = models["model"].predict(X_sample)
    cal_scores = models["calibrator"].predict(raw_scores)

    return idx, y_sample, cal_scores


def run_agent_batch(
    idx: np.ndarray,
    cal_scores: np.ndarray,
    y_sample: np.ndarray,
    raw_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    tool_mode: str,
    score_to_label_fn: Callable[[float], str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Run the full 4-agent pipeline for each sampled transaction.

    hist_df must be the train+val historical data — it is passed to the
    investigation tools so they never touch test-set labels.
    progress_callback(completed, total) is called after each transaction.
    Returns rows suitable for pd.DataFrame().
    """
    rows = []
    total = len(idx)
    for i, (sample_i, cal_score, y_val) in enumerate(zip(idx, cal_scores, y_sample)):
        txn_raw = raw_df.iloc[sample_i]
        ml_out = {"fraud_score": float(cal_score), "shap_values": None, "feature_importance": []}
        try:
            tool_res = investigate_transaction(
                txn_raw, hist_df, fraud_score=float(cal_score), mode=tool_mode
            )
            assessment = run_agents(
                transaction=txn_raw.to_dict(), ml_output=ml_out, tool_results=tool_res
            )
            rows.append({
                "fraud_score":          float(cal_score),
                "actual":               int(y_val),
                "threshold_decision":   score_to_label_fn(float(cal_score)),
                "agent_decision":       assessment.decision or "UNKNOWN",
                "confidence":           assessment.confidence,
                "risk_level":           assessment.risk_level or "UNKNOWN",
                "risk_score":           assessment.risk_score,
                "latency_ms":           assessment.execution_time_ms,
            })
        except Exception:
            rows.append({
                "fraud_score":          float(cal_score),
                "actual":               int(y_val),
                "threshold_decision":   score_to_label_fn(float(cal_score)),
                "agent_decision":       "ERROR",
                "confidence":           0.0,
                "risk_level":           "UNKNOWN",
                "risk_score":           0.0,
                "latency_ms":           0.0,
            })
        if progress_callback:
            progress_callback(i + 1, total)
    return rows


def compute_batch_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    score_to_label_fn: Callable[[float], str],
) -> dict:
    """Compute all evaluation metrics from y_true and calibrated scores.

    Returns a flat dict consumed directly by the rendering layer.
    """
    predicted_labels = np.array([score_to_label_fn(s) for s in scores])
    flagged = np.isin(predicted_labels, ["BLOCK", "REVIEW"]).astype(int)

    has_both_classes = len(np.unique(y_true)) > 1

    auc = roc_auc_score(y_true, scores) if has_both_classes else float("nan")
    ap  = average_precision_score(y_true, scores) if has_both_classes else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if has_both_classes:
        fpr, tpr, _          = roc_curve(y_true, scores)
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, scores)
    else:
        fpr = tpr = np.array([0.0, 1.0])
        prec_arr = rec_arr = np.array([0.0, 1.0])

    thresholds = np.linspace(0.1, 0.99, 90)
    sweep_rows = []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        _, fp_t, fn_t, tp_t = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        sweep_rows.append({"threshold": thr, "precision": p, "recall": r, "f1": f, "flag_rate": pred.mean()})
    sweep_df = pd.DataFrame(sweep_rows)

    return {
        "auc": auc, "ap": ap,
        "precision": precision, "recall": recall, "f1": f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fpr, "tpr": tpr,
        "prec_arr": prec_arr, "rec_arr": rec_arr,
        "sweep_df": sweep_df,
        "predicted_labels": predicted_labels,
    }
