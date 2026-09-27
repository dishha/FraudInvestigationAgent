"""
Fraud Detection System — Streamlit UI
All rendering lives here. Processing logic lives in helpers/ and scripts/.

Run with:
    streamlit run app.py
"""

import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents import investigator
from config import (
    BLOCK_THRESHOLD,
    DECISION_COLOR,
    LOG_DIR,
    LOG_FILE,
    ML_FASTPATH_APPROVE,
    ML_FASTPATH_BLOCK,
    REVIEW_THRESHOLD,
    SOFT_DECLINE_THRESHOLD,
    score_to_label,
)
from helpers.batch_runner import compute_batch_metrics, run_agent_batch, run_ml_batch
from helpers.decision_strategy import (
    extract_decision_keyword,
    fallback_explanation_decision,
    merge_agent_and_threshold_decision,
    normalize_decision_label,
    threshold_decision_from_score,
)
from helpers.loaders import load_data, load_feature_importance, load_models
from helpers.rules_engine import HardRulesEngine
from scripts.feature_engineering import transform_features_for_scoring
from scripts.fraud_investigation_agents import investigate_transaction, run_agents
from scripts.ml_pipeline import score_with_ml

# ── Page config (must be the first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger(__name__)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #ffffff; }
    .main { padding: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    .success-box { background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border: 1px solid #c3e6cb; }
    .error-box   { background-color: #f8d7da; padding: 1rem; border-radius: 0.5rem; border: 1px solid #f5c6cb; }
    .decision-banner { padding: 12px 16px; border-radius: 8px; border: 1px solid; font-size: 14px; font-weight: 500; margin: 8px 0; }
    .pipeline-step { border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; text-align: center; font-size: 12px; }
    .step-badge-done { background: #EAF3DE; color: #3B6D11; border-radius: 10px; padding: 2px 8px; font-size: 10px; }
    .step-badge-flagged { background: #FAEEDA; color: #633806; border-radius: 10px; padding: 2px 8px; font-size: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.models = None
    st.session_state.data = None
    st.session_state.history = []
    st.session_state.error_message = None

# Decision banner color map
_BANNER_STYLES = {
    "BLOCK":        ("FCEBEB", "F7C1C1", "A32D2D"),
    "REVIEW":       ("FAEEDA", "FAC775", "633806"),
    "SOFT_DECLINE": ("E6F1FB", "B5D4F4", "185FA5"),
    "APPROVE":      ("EAF3DE", "C0DD97", "3B6D11"),
}

# Analytics pie color map
_PIE_COLORS = {
    "BLOCK": "#E24B4A",
    "REVIEW": "#EF9F27",
    "SOFT_DECLINE": "#378ADD",
    "APPROVE": "#639922",
}


def _decision_banner(decision_keyword: str, fraud_score: float) -> None:
    bg, border, text = _BANNER_STYLES.get(decision_keyword, ("F0F2F6", "CCCCCC", "333333"))
    explanation = fallback_explanation_decision(decision_keyword, fraud_score)
    st.markdown(
        f'<div class="decision-banner" style="background:#{bg};border-color:#{border};color:#{text};">'
        f'<strong>{decision_keyword}</strong> — {explanation}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_pipeline(assessment, decision_keyword: str) -> None:
    findings = assessment.investigation_findings or {}
    finding_note = (
        str(findings.get("summary", findings.get("verdict", "Investigation complete")))[:60]
        if findings else "Investigation complete"
    )
    steps = [
        {
            "name": "ML Analyst",
            "note": f"Fraud score: {assessment.risk_score:.0%}" if assessment.risk_score else "Scoring complete",
            "flagged": False,
        },
        {
            "name": "Investigator",
            "note": finding_note,
            "flagged": False,
        },
        {
            "name": "Risk Assessor",
            "note": f"Risk: {assessment.risk_level}" if assessment.risk_level else "Assessment complete",
            "flagged": assessment.risk_level in ("HIGH", "CRITICAL"),
        },
        {
            "name": "Explainer",
            "note": f"Decision: {decision_keyword}",
            "flagged": decision_keyword in ("BLOCK", "REVIEW"),
        },
    ]
    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        badge_class = "step-badge-flagged" if step["flagged"] else "step-badge-done"
        badge_text = "flagged" if step["flagged"] else "done"
        col.markdown(
            f'<div class="pipeline-step">'
            f'<div style="font-weight:600;margin-bottom:4px;">{step["name"]}</div>'
            f'<div style="color:#666;margin-bottom:6px;font-size:11px;">{step["note"]}</div>'
            f'<span class="{badge_class}">{badge_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# SIDEBAR
# ============================================================================

def _render_sidebar(models: dict) -> str:
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            '<span style="width:12px;height:12px;background:#E24B4A;border-radius:50%;'
            'display:inline-block;flex-shrink:0;"></span>'
            '<span style="font-size:18px;font-weight:700;">Fraud Intelligence Engine</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.subheader("Tool Selection Mode")
        tool_mode = st.radio(
            "Tool selection strategy",
            ["adaptive", "all"],
            format_func=lambda m: "Adaptive (score-gated)" if m == "adaptive" else "All tools (always)",
            help="Adaptive skips tools when the ML score is already high/low confidence.",
        )
        auc_str = f"{models['auc']:.3f}" if models.get("auc") else "N/A"
        st.markdown("**System Status**", unsafe_allow_html=False)
        st.markdown(f"""
<div style="font-size:13px; line-height:2; color:var(--text-color);">
  <div>🟢 &nbsp;Models &nbsp;<span style="color:#6b7280;">ready</span></div>
  <div>🟢 &nbsp;Data &nbsp;<span style="color:#6b7280;">ready</span></div>
  <div style="margin-top:8px; color:#6b7280; font-size:12px;">
    AUC <strong style="color:inherit;">{auc_str}</strong> · LightGBM
  </div>
  <div style="color:#6b7280; font-size:12px;">
    Scored &nbsp;<strong style="color:inherit;">{len(st.session_state.history)}</strong> transactions
  </div>
</div>
""", unsafe_allow_html=True)
    return tool_mode


# ============================================================================
# TAB 1 — SCORE TRANSACTION
# ============================================================================

def _tab_score_transaction(data: dict, models: dict, tool_mode: str, rules_engine: HardRulesEngine) -> None:
    auc_str = f"{models['auc']:.3f}" if models.get("auc") else "N/A"
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:12px; padding:10px 0 16px; '
        f'border-bottom:1px solid #e5e7eb; margin-bottom:16px;">'
        f'<span style="font-size:13px; font-weight:600; color:#111;">Fraud Detection</span>'
        f'<span style="font-size:13px; color:#9ca3af;">/</span>'
        f'<span style="font-size:13px; color:#6b7280;">Score transaction</span>'
        f'<span style="background:#dcfce7; color:#166534; font-size:11px; font-weight:500;'
        f'padding:2px 10px; border-radius:20px; margin-left:4px;">{tool_mode}</span>'
        f'<span style="margin-left:auto; font-size:12px; color:#9ca3af;">'
        f'LightGBM · v1.0 · {auc_str} AUC</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if "test_idx" not in st.session_state:
        st.session_state.test_idx = 0

    col1, _, col3 = st.columns([2, 1, 1])
    with col1:
        st.session_state.test_idx = st.number_input(
            "Select transaction index",
            min_value=0, max_value=len(data["X_test"]) - 1,
            value=st.session_state.test_idx, step=1,
        )
    with col3:
        if st.button("🔄 Random", key="random_btn"):
            st.session_state.test_idx = np.random.randint(0, len(data["X_test"]) - 1)
            st.rerun()

    test_idx = st.session_state.test_idx
    X_test = transform_features_for_scoring(data["X_test"], models["feature_names"])
    transaction_features = X_test.iloc[test_idx : test_idx + 1]
    transaction_raw = data["test_raw"].iloc[test_idx]
    actual_label = data["y_test"].iloc[test_idx]

    c1, c2, c3, c4 = st.columns(4)
    amount = transaction_raw.get("TransactionAmt", 0)
    c1.metric("Amount", f"${amount:.2f}")
    c2.metric("Card (Last 4)", str(transaction_raw.get("card1", "N/A"))[-4:])
    c3.metric("Actual Label", "FRAUD" if actual_label == 1 else "LEGIT")
    c4.metric("Index", test_idx)
    st.write("")

    st.markdown("""<style>
div[data-testid="stButton"] > button[kind="primaryFormSubmit"],
div[data-testid="stButton"] > button {
  background-color: #111827 !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em !important;
}
div[data-testid="stButton"] > button:hover {
  background-color: #374151 !important;
}
</style>""", unsafe_allow_html=True)
    if st.button("Score transaction", use_container_width=True, type="primary"):
        logger.info("Scoring transaction %d", test_idx)
        with st.spinner("Scoring transaction..."):
            try:
                import time as _time
                _tx_start = _time.time()

                # Layer 0: Hard Rules Engine — runs before ML, cannot be overridden
                rule_result = rules_engine.evaluate(transaction_raw)

                # Layer 1: ML scoring (always runs for the fraud score display)
                ml_output = score_with_ml(transaction_features, models)
                fraud_score = ml_output["fraud_score"]

                ml_fastpath = None  # set by Layer 1 if score is in a confident tail
                threshold_overrode_agent = False

                if rule_result.triggered:
                    # Layer 0 — Hard rule fires; decision is final
                    logger.warning(
                        "Hard rule triggered: %s | %s",
                        rule_result.rule_name, rule_result.evidence,
                    )
                    final_decision_label = "🔴 BLOCK"
                    decision_keyword = "BLOCK"
                    decision_label = final_decision_label
                    decision = final_decision_label
                    risk_level = "CRITICAL"
                    confidence = 1.0
                    risk_score = 1.0
                    tool_results = {}
                    assessment = None
                else:
                    rule_result = None

                    # Layer 1 — ML Score Fast-Path: skip agents for confident tails
                    # "all tools" mode bypasses the approve fast-path so agents always run.
                    if fraud_score >= ML_FASTPATH_BLOCK:
                        ml_fastpath = f"score {fraud_score:.1%} ≥ {ML_FASTPATH_BLOCK:.0%} — agents skipped"
                        logger.info("ML fast-path BLOCK: %.3f", fraud_score)
                        final_decision_label = "\U0001f534 BLOCK"
                        decision_keyword = "BLOCK"
                        decision_label = final_decision_label
                        decision = final_decision_label
                        risk_level = "CRITICAL"
                        confidence = 0.95
                        risk_score = fraud_score
                        tool_results = {}
                        assessment = None
                    elif fraud_score < ML_FASTPATH_APPROVE and tool_mode != "all":
                        ml_fastpath = f"score {fraud_score:.1%} < {ML_FASTPATH_APPROVE:.0%} — agents skipped"
                        logger.info("ML fast-path APPROVE: %.3f", fraud_score)
                        final_decision_label = "\U0001f7e2 APPROVE"
                        decision_keyword = "APPROVE"
                        decision_label = final_decision_label
                        decision = final_decision_label
                        risk_level = "LOW"
                        confidence = 0.95
                        risk_score = fraud_score
                        tool_results = {}
                        assessment = None
                    else:
                        # Layers 2–3: agentic investigation for scores in 0.15–0.92
                        tool_results = investigate_transaction(
                            transaction_raw, data["hist_raw"],
                            fraud_score=fraud_score, mode=tool_mode,
                        )
                        assessment = run_agents(
                            transaction=transaction_raw.to_dict(),
                            ml_output=ml_output,
                            tool_results=tool_results,
                        )

                        decision = assessment.decision
                        confidence = assessment.confidence
                        risk_level = assessment.risk_level
                        risk_score = assessment.risk_score

                        decision_label = normalize_decision_label(decision)
                        threshold_label = threshold_decision_from_score(
                            fraud_score, BLOCK_THRESHOLD, REVIEW_THRESHOLD, SOFT_DECLINE_THRESHOLD,
                        )
                        final_decision_label = merge_agent_and_threshold_decision(decision_label, threshold_label)
                        decision_keyword = extract_decision_keyword(final_decision_label)

                        threshold_overrode_agent = final_decision_label != decision_label
                        if threshold_overrode_agent:
                            logger.warning(
                                "Threshold override: agent=%s threshold=%s final=%s",
                                decision_label, threshold_label, final_decision_label,
                            )

                        if decision is None or extract_decision_keyword(decision_label) not in {"APPROVE", "SOFT_DECLINE", "REVIEW", "BLOCK"}:
                            logger.error("Agent failure — falling back to threshold decision")
                            final_decision_label = threshold_label
                            decision_keyword = final_decision_label.split()[-1]
                            risk_level, confidence = {
                                "BLOCK":        ("CRITICAL", 0.95),
                                "REVIEW":       ("HIGH",     0.90),
                                "SOFT_DECLINE": ("MEDIUM",   0.80),
                            }.get(decision_keyword, ("LOW", 0.85))

                        decision_label = final_decision_label
                        decision = final_decision_label

                _total_ms = (_time.time() - _tx_start) * 1000
                _latency = assessment.execution_time_ms if assessment else _total_ms
                _cost = assessment.agent_cost_usd if assessment else 0.0
                st.session_state.history.append({
                    "timestamp":      datetime.now(),
                    "transaction_id": test_idx,
                    "fraud_score":    fraud_score,
                    "decision":       decision_keyword,
                    "risk_level":     risk_level,
                    "risk_score":     risk_score,
                    "latency_ms":     _latency,
                    "amount":         amount,
                    "actual":         actual_label,
                })
                logger.info("Decision: %s | Risk: %s | %.1f ms | cost $%.4f", decision, risk_level, _latency, _cost)

                st.subheader("Assessment Results")
                res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
                res_col1.metric("Fraud score", f"{fraud_score:.1%}")
                res_col2.metric("Risk level", risk_level)
                res_col3.metric("Confidence", f"{confidence:.0%}")
                res_col4.metric("Latency", f"{_latency:.1f} ms")
                res_col5.metric("Agent cost", f"${_cost:.4f}")
                _is_correct = (actual_label == 1) == (decision.split()[-1] in {"BLOCK", "REVIEW"})
                _badge_color = "#dcfce7" if _is_correct else "#fee2e2"
                _badge_text_color = "#166534" if _is_correct else "#991b1b"
                _badge_label = "Correct prediction" if _is_correct else "Wrong prediction"
                st.markdown(f"""
<div style="display:inline-block; background:{_badge_color};
  color:{_badge_text_color}; font-size:12px; font-weight:500;
  padding:4px 14px; border-radius:20px; margin:4px 0 12px;">
  {"✓" if _is_correct else "✗"} &nbsp;{_badge_label}
</div>
""", unsafe_allow_html=True)
                if rule_result:
                    st.markdown(
                        f'<div style="background:#FEF3C7; border:1px solid #F59E0B; border-radius:8px; '
                        f'padding:12px 16px; margin:8px 0 4px; font-size:13px;">'
                        f'<strong>&#9889; Hard Rule Triggered &mdash; {rule_result.rule_name}</strong><br>'
                        f'{rule_result.evidence}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif ml_fastpath:
                    st.markdown(
                        f'<div style="background:#EFF6FF; border:1px solid #93C5FD; border-radius:8px; '
                        f'padding:12px 16px; margin:8px 0 4px; font-size:13px;">'
                        f'<strong>&#9889; ML Fast-Path</strong> &mdash; {ml_fastpath}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                _decision_banner(decision_keyword, fraud_score)

                if ml_output.get("shap_values") is not None:
                    shap_df = pd.DataFrame({
                        "feature": models["feature_names"],
                        "shap_value": ml_output["shap_values"][0],
                    }).sort_values("shap_value", key=abs, ascending=False).head(8)

                    shap_col, _ = st.columns([3, 1])
                    with shap_col:
                        fig = px.bar(
                            shap_df, x="shap_value", y="feature", orientation="h",
                            color="shap_value",
                            color_continuous_scale=["#E24B4A", "#f5f5f5", "#378ADD"],
                            color_continuous_midpoint=0,
                            labels={"shap_value": "SHAP contribution", "feature": ""},
                        )
                        fig.update_layout(
                            height=260,
                            margin=dict(l=0, r=0, t=28, b=0),
                            title=dict(text="Why this score?", font=dict(size=13), x=0),
                            coloraxis_showscale=False,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(tickfont=dict(size=11)),
                            xaxis=dict(tickfont=dict(size=11), title_font=dict(size=11)),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                st.subheader("Decision Explanation")
                if rule_result:
                    st.error(rule_result.evidence)
                elif ml_fastpath:
                    st.info(fallback_explanation_decision(decision_keyword, fraud_score))
                else:
                    explanation = assessment.customer_explanation or ""
                    if explanation.strip() and not threshold_overrode_agent:
                        st.info(explanation)
                    else:
                        st.info(fallback_explanation_decision(decision_keyword, fraud_score))

                st.subheader("Agent Pipeline")
                if rule_result:
                    st.info("Agent investigation bypassed — hard rule decision is final.")
                elif ml_fastpath:
                    st.info(f"Agent investigation bypassed — ML Fast-Path ({ml_fastpath}).")
                else:
                    _render_pipeline(assessment, decision_keyword)

                with st.expander("Raw diagnostics", expanded=False):
                    if rule_result:
                        st.write("**Decision path:**", "HARD RULE (Layer 0)")
                        st.write("**Rule:**", rule_result.rule_name)
                        st.write("**Evidence:**", rule_result.evidence)
                        st.write("**ML output**"); st.json(ml_output)
                    elif ml_fastpath:
                        st.write("**Decision path:**", "ML FAST-PATH (Layer 1)")
                        st.write("**Reason:**", ml_fastpath)
                        st.write("**ML output**"); st.json(ml_output)
                    else:
                        st.write("**Agent execution mode:**", "MOCK" if investigator.MOCK_MODE else "REAL")
                        st.write("**Agent recommendation:**", str(assessment.decision))
                        st.write("**Displayed decision:**", decision_label)
                        st.metric("Latency", f"{assessment.execution_time_ms:.1f} ms")
                        st.write("**ML output**");      st.json(ml_output)
                        st.write("**Investigation findings**"); st.json(assessment.investigation_findings)
                        st.write("**Risk assessment**");       st.json(assessment.risk_assessment)
                        st.write("**Agent reasoning**");       st.json(assessment.reasoning_details)
                        if investigator.MOCK_MODE:
                            st.warning("Mock mode — agent responses are simulated.")

            except Exception as exc:
                logger.error("Error scoring: %s", exc)
                st.error(f"Error: {exc}")


# ============================================================================
# TAB 2 — BATCH EVALUATION
# ============================================================================

def _tab_batch_evaluation(data: dict, models: dict, tool_mode: str) -> None:
    st.title("📈 Batch Evaluation")

    max_ml = min(2000, len(data["X_test"]))
    agent_cap = 100

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        include_agents = st.checkbox(
            "Include agent reasoning",
            value=False,
            help=(
                "Runs the full 4-agent pipeline per transaction. "
                f"Capped at {agent_cap} samples. Fast in mock mode."
            ),
        )
        max_samples = agent_cap if include_agents else max_ml
        n_samples = st.slider(
            "Sample size",
            min_value=10, max_value=max_samples,
            value=min(500, max_samples),
            step=10 if include_agents else 50,
        )
        if include_agents:
            mode_label = "Adaptive (score-gated)" if tool_mode == "adaptive" else "All tools"
            st.caption(
                f"Agent mode on — {n_samples} transactions, tool mode: **{mode_label}**. "
                "Runs in mock mode when no Anthropic key is set."
            )
    with col_ctrl2:
        st.write("")
        st.write("")
        run_eval = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    if run_eval or "batch_results" in st.session_state:
        if run_eval:
            with st.spinner(f"ML scoring {n_samples} transactions…"):
                idx, y_sample, cal_scores = run_ml_batch(data, models, n_samples)

            agent_rows = None
            if include_agents:
                progress_bar = st.progress(0, text="Running agents…")

                def _update(completed: int, total: int) -> None:
                    progress_bar.progress(completed / total, text=f"Running agents… {completed}/{total}")

                agent_rows = run_agent_batch(
                    idx, cal_scores, y_sample,
                    raw_df=data["test_raw"],
                    hist_df=data["hist_raw"],
                    tool_mode=tool_mode,
                    score_to_label_fn=score_to_label,
                    progress_callback=_update,
                )
                progress_bar.empty()

            st.session_state.batch_results = {
                "y_true": y_sample,
                "scores": cal_scores,
                "n": n_samples,
                "agent_rows": agent_rows,
            }

        br = st.session_state.batch_results
        y_true, scores = br["y_true"], br["scores"]
        agent_rows = br.get("agent_rows")

        m = compute_batch_metrics(y_true, scores, score_to_label)

        # ── Key metrics ───────────────────────────────────────────────────────
        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AUC-ROC",       f"{m['auc']:.3f}")
        c2.metric("Avg Precision", f"{m['ap']:.3f}")
        c3.metric("Precision",     f"{m['precision']:.3f}")
        c4.metric("Recall",        f"{m['recall']:.3f}")
        c5.metric("F1",            f"{m['f1']:.3f}")

        # ── Score distribution (hero chart) ──────────────────────────────────
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=scores[y_true == 0], name="Legitimate",
                                        opacity=0.65, marker_color="#2196F3", nbinsx=40))
        fig_dist.add_trace(go.Histogram(x=scores[y_true == 1], name="Fraud",
                                        opacity=0.65, marker_color="#F44336", nbinsx=40))
        for thr, label, color in [
            (BLOCK_THRESHOLD, "BLOCK", "red"),
            (REVIEW_THRESHOLD, "REVIEW", "orange"),
            (SOFT_DECLINE_THRESHOLD, "SOFT_DECLINE", "gold"),
        ]:
            fig_dist.add_vline(x=thr, line_dash="dash", line_color=color,
                               annotation_text=label, annotation_position="top right")
        fig_dist.update_layout(barmode="overlay", title="Calibrated Fraud Score by Actual Class",
                               xaxis_title="Fraud Score", yaxis_title="Count", height=320,
                               margin=dict(t=40, b=0))
        st.plotly_chart(fig_dist, use_container_width=True)

        # ── ROC + confusion matrix ────────────────────────────────────────────
        col_roc, col_cm = st.columns(2)
        with col_roc:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"], mode="lines",
                                         name=f"AUC={m['auc']:.3f}", line=dict(color="#4CAF50", width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         name="Random", line=dict(color="grey", dash="dash")))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=320,
                                  xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                                  margin=dict(t=40, b=0))
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_cm:
            cm_data = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
            fig_cm = go.Figure(go.Heatmap(
                z=cm_data, x=["Pred: Legit", "Pred: Flagged"],
                y=["Actual: Legit", "Actual: Fraud"],
                colorscale="Blues", showscale=False,
                text=[[str(v) for v in row] for row in cm_data],
                texttemplate="%{text}", textfont={"size": 20},
            ))
            fig_cm.update_layout(title="Confusion Matrix", height=320, margin=dict(t=40, b=0))
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── Secondary analysis (collapsed by default) ─────────────────────────
        with st.expander("Precision-Recall curve · Threshold sweep · Decision breakdown"):
            col_pr, col_sweep = st.columns(2)
            with col_pr:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=m["rec_arr"], y=m["prec_arr"], mode="lines",
                                            name=f"AP={m['ap']:.3f}", line=dict(color="#9C27B0", width=2)))
                base = y_true.mean()
                fig_pr.add_hline(y=base, line_dash="dash", line_color="grey",
                                 annotation_text=f"Baseline ({base:.2f})")
                fig_pr.update_layout(title="Precision-Recall", xaxis_title="Recall",
                                     yaxis_title="Precision", height=300,
                                     xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig_pr, use_container_width=True)

            with col_sweep:
                sweep = m["sweep_df"]
                fig_sweep = go.Figure()
                fig_sweep.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["precision"], mode="lines", name="Precision"))
                fig_sweep.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["recall"],    mode="lines", name="Recall"))
                fig_sweep.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["f1"],        mode="lines", name="F1",
                                               line=dict(width=2, dash="dot")))
                fig_sweep.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["flag_rate"], mode="lines",
                                               name="Flag rate", line=dict(dash="dash")))
                for thr_val, thr_name in [
                    (BLOCK_THRESHOLD, "BLOCK"), (REVIEW_THRESHOLD, "REVIEW"), (SOFT_DECLINE_THRESHOLD, "SOFT_DECLINE"),
                ]:
                    fig_sweep.add_vline(x=thr_val, line_dash="dash", line_color="grey", annotation_text=thr_name)
                fig_sweep.update_layout(title="Threshold Sweep", xaxis_title="Threshold",
                                        yaxis=dict(range=[0, 1]), height=300)
                st.plotly_chart(fig_sweep, use_container_width=True)

            label_counts = pd.Series(m["predicted_labels"]).value_counts().reset_index()
            label_counts.columns = ["Decision", "Count"]
            label_counts["% of sample"] = (label_counts["Count"] / len(m["predicted_labels"]) * 100).round(1)
            st.dataframe(label_counts, use_container_width=True, hide_index=True)

        # ── Agent section ─────────────────────────────────────────────────────
        if agent_rows:
            agent_df = pd.DataFrame(agent_rows)
            st.divider()
            st.subheader("Agent Results")

            valid = agent_df[agent_df["agent_decision"] != "ERROR"]
            agree = (valid["agent_decision"] == valid["threshold_decision"]).mean()
            avg_conf = valid["confidence"].mean()
            avg_lat = valid["latency_ms"].mean()
            flagged_agent = valid["agent_decision"].isin(["BLOCK", "REVIEW"])
            flagged_true = valid["actual"] == 1
            a_prec = (flagged_agent & flagged_true).sum() / max(flagged_agent.sum(), 1)
            a_rec  = (flagged_agent & flagged_true).sum() / max(flagged_true.sum(), 1)
            a_f1   = 2 * a_prec * a_rec / (a_prec + a_rec) if (a_prec + a_rec) > 0 else 0.0

            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("Agent Precision", f"{a_prec:.3f}")
            a2.metric("Agent Recall",    f"{a_rec:.3f}")
            a3.metric("Agent F1",        f"{a_f1:.3f}")
            a4.metric("Avg Confidence",  f"{avg_conf:.0%}")
            a5.metric("Avg Latency",     f"{avg_lat:.0f} ms")
            st.caption(
                f"Agent–threshold agreement: **{agree:.0%}** of {len(valid)} transactions · "
                f"**{'MOCK' if investigator.MOCK_MODE else 'REAL'}** mode"
            )

            col_adec, col_agree = st.columns(2)
            with col_adec:
                dec_counts = agent_df["agent_decision"].value_counts().reset_index()
                dec_counts.columns = ["Decision", "Count"]
                fig_adec = go.Figure(go.Bar(
                    x=dec_counts["Decision"], y=dec_counts["Count"],
                    marker_color=[DECISION_COLOR.get(d, "#9E9E9E") for d in dec_counts["Decision"]],
                ))
                fig_adec.update_layout(title="Agent Decisions", xaxis_title="Decision",
                                       yaxis_title="Count", height=300, margin=dict(t=40, b=0))
                st.plotly_chart(fig_adec, use_container_width=True)

            with col_agree:
                labels = ["APPROVE", "SOFT_DECLINE", "REVIEW", "BLOCK"]
                cross = pd.crosstab(agent_df["threshold_decision"], agent_df["agent_decision"])
                cross = cross.reindex(index=labels, columns=labels, fill_value=0)
                fig_cross = go.Figure(go.Heatmap(
                    z=cross.values.tolist(), x=cross.columns.tolist(), y=cross.index.tolist(),
                    colorscale="Blues", showscale=True,
                    text=[[str(v) for v in row] for row in cross.values.tolist()],
                    texttemplate="%{text}",
                ))
                fig_cross.update_layout(title="Agent vs Threshold", xaxis_title="Agent decision",
                                        yaxis_title="Threshold decision", height=300, margin=dict(t=40, b=0))
                st.plotly_chart(fig_cross, use_container_width=True)

            with st.expander("Score correlation · Latency · Confidence"):
                col_scatter, col_lat = st.columns(2)
                with col_scatter:
                    fig_scat = px.scatter(
                        agent_df, x="fraud_score", y="risk_score",
                        color="agent_decision", color_discrete_map=DECISION_COLOR,
                        symbol="actual", symbol_map={0: "circle", 1: "x"},
                        title="ML Score vs Agent Risk Score",
                        labels={"fraud_score": "ML Fraud Score", "risk_score": "Agent Risk Score",
                                "agent_decision": "Decision", "actual": "Actual"},
                        height=320,
                    )
                    fig_scat.update_traces(marker=dict(size=8, opacity=0.75))
                    st.plotly_chart(fig_scat, use_container_width=True)

                with col_lat:
                    fig_lat = px.histogram(
                        agent_df[agent_df["latency_ms"] > 0],
                        x="latency_ms", color="agent_decision",
                        color_discrete_map=DECISION_COLOR, nbins=30,
                        title="Latency Distribution",
                        labels={"latency_ms": "Latency (ms)", "agent_decision": "Decision"},
                        height=320,
                    )
                    fig_lat.update_layout(barmode="overlay")
                    st.plotly_chart(fig_lat, use_container_width=True)

                fig_box = px.box(
                    agent_df[agent_df["agent_decision"] != "ERROR"],
                    x="agent_decision", y="confidence",
                    color="agent_decision", color_discrete_map=DECISION_COLOR,
                    points="all", title="Confidence by Decision",
                    labels={"agent_decision": "Decision", "confidence": "Confidence"},
                    height=300,
                )
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.info("Set a sample size and click **Run Evaluation** to start.")


# ============================================================================
# TAB 3 — ANALYTICS
# ============================================================================

def _tab_analytics() -> None:
    st.title("📊 Analytics Dashboard")
    if not st.session_state.history:
        st.info("💡 No transactions scored yet. Go to 'Score Transaction' to get started!")
        return

    history_df = pd.DataFrame(st.session_state.history)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", len(history_df))
    c2.metric("Avg Fraud Score", f"{history_df['fraud_score'].mean():.1%}")
    c3.metric("Blocked", len(history_df[history_df["decision"] == "BLOCK"]))
    accuracy = (
        (history_df["decision"].isin(["BLOCK", "REVIEW"]) == (history_df["actual"] == 1)).sum()
        / len(history_df) * 100
    )
    c4.metric("Accuracy", f"{accuracy:.1f}%")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        decision_counts = history_df["decision"].value_counts()
        st.plotly_chart(
            px.pie(
                values=decision_counts.values,
                names=decision_counts.index,
                title="Decision Distribution",
                color=decision_counts.index,
                color_discrete_map=_PIE_COLORS,
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            px.histogram(history_df, x="fraud_score", nbins=20, title="Fraud Score Distribution"),
            use_container_width=True,
        )
    st.divider()
    st.subheader("Transaction History")
    display_df = history_df[["timestamp", "transaction_id", "fraud_score", "decision", "risk_level", "latency_ms", "amount"]].copy()
    display_df["fraud_score"] = display_df["fraud_score"].apply(lambda x: f"{x:.1%}")
    display_df["latency_ms"]  = display_df["latency_ms"].apply(lambda x: f"{x:.1f} ms")
    display_df["amount"]      = display_df["amount"].apply(lambda x: f"${x:.2f}")
    display_df["timestamp"]   = display_df["timestamp"].apply(lambda x: x.strftime("%H:%M:%S"))
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================================
# TAB 4 — LOGS
# ============================================================================

def _tab_logs() -> None:
    st.title("📝 System Logs")
    if not LOG_FILE.exists():
        st.info("No logs yet. Run a transaction to generate logs.")
        return

    with open(LOG_FILE) as f:
        logs = f.read()

    col1, col2 = st.columns(2)
    with col1:
        log_level = st.selectbox("Filter by level", ["ALL", "INFO", "ERROR", "WARNING"])
    with col2:
        lines_to_show = st.slider("Lines to show", 10, 500, 100)

    log_lines = logs.split("\n")
    if log_level != "ALL":
        log_lines = [l for l in log_lines if log_level in l]
    st.code("\n".join(log_lines[-lines_to_show:]), language="text")
    st.download_button("📥 Download Full Log", data=logs, file_name="fraud_detection.log", mime="text/plain")


# ============================================================================
# TAB 5 — SETTINGS
# ============================================================================

def _tab_settings(data: dict, models: dict, importance_df: pd.DataFrame) -> None:
    st.title("⚙️ Settings")
    st.subheader("System Information")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Version:** 1.0.0")
        st.write("**Framework:** Streamlit")
        st.write("**Model:** LightGBM")
    with c2:
        st.write(f"**Test Samples:** {len(data['X_test']):,}")
        st.write(f"**Features:** {len(models['feature_names'])}")
        st.write(f"**Tree Count:** {models['model'].num_trees()}")
    st.divider()
    st.subheader("File Status")
    files = {
        "Model":        "model/lightgbm_model.pkl",
        "Calibrator":   "model/calibrator.pkl",
        "Features":     "model/feature_names.pkl",
        "Importance":   "model/feature_importance.csv",
        "Test Data":    "data/features_test.pkl",
        "Prepared Data":"data/ieee_prepared.pkl",
    }
    for name, path in files.items():
        if Path(path).exists():
            st.success(f"✅ {name}: {path}")
        else:
            st.error(f"❌ {name}: {path} (not found)")
    st.divider()
    auc_str = f"{models['auc']:.3f}" if models.get("auc") else "N/A"
    st.subheader("About")
    st.write(f"""
    Production-grade fraud detection system:
    - LightGBM model trained on IEEE-CIS dataset ({auc_str} AUC)
    - 4-agent reasoning pipeline (MLAnalyst → TransactionInvestigator → RiskAssessor → DecisionExplainer)
    - Adaptive tool selection with latency budgeting
    - Real-time scoring and batch evaluation
    """)

    if not importance_df.empty:
        st.divider()
        st.subheader("Model feature importance")
        fig_imp = px.bar(
            importance_df.head(15), x="importance", y="feature",
            orientation="h",
            labels={"importance": "Importance", "feature": ""},
        )
        fig_imp.update_layout(height=380, margin=dict(l=0, r=0, t=8, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    models_result = load_models()
    data_result = load_data()
    importance_df = load_feature_importance()

    if not models_result["success"]:
        st.error(f"❌ Error loading models: {models_result['error']}")
        st.code("model/lightgbm_model.pkl\nmodel/calibrator.pkl\nmodel/feature_names.pkl")
        return
    if not data_result["success"]:
        st.error(f"❌ Error loading data: {data_result['error']}")
        st.code("data/features_test.pkl\ndata/ieee_prepared.pkl")
        return
    if importance_df is None:
        st.warning("⚠️ Feature importance file not found")
        importance_df = pd.DataFrame()

    models = models_result
    data = data_result
    rules_engine = HardRulesEngine(data["hist_raw"])
    logger.info("Application started")

    tool_mode = _render_sidebar(models)

    tabs = st.tabs(["Score Transaction", "Batch Evaluation", "Analytics", "Logs", "Settings"])
    with tabs[0]:
        _tab_score_transaction(data, models, tool_mode, rules_engine)
    with tabs[1]:
        _tab_batch_evaluation(data, models, tool_mode)
    with tabs[2]:
        _tab_analytics()
    with tabs[3]:
        _tab_logs()
    with tabs[4]:
        _tab_settings(data, models, importance_df)


if __name__ == "__main__":
    main()
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.caption("🚀 Fraud Detection System")
    c2.caption(f"📊 {len(st.session_state.history)} scored")
    c3.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
