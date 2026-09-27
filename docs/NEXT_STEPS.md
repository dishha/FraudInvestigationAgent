# Next Steps

## 1. Cost tracking per transaction run

Add API cost (input + output tokens × model rate) to each transaction that goes through the agent pipeline.

**What to do:**
- Capture `usage.input_tokens` and `usage.output_tokens` from each Claude API response in `agents/investigator.py`
- Sum across all 4 agents (MLAnalyst, TransactionInvestigator, RiskAssessor, DecisionExplainer)
- Add `agent_cost_usd` field to `FraudAssessment`
- Display in the dashboard under Assessment Results
- Log alongside decision and latency

**Scope:** Only applies when agents run (not ML fast-path or hard rule path). Show `—` for fast-pathed transactions.

---

## 2. Agent Evaluation

### 2a. Per-layer confusion matrix in batch tab

Split batch results into zones and show precision/recall separately for each:

| Layer | Zone | Metric to show |
|---|---|---|
| Hard Rules | triggered transactions | precision, false positive rate |
| ML Fast-path approve | score < `fastpath_approve` | false negative rate (fraud that slipped through) |
| ML Fast-path block | score ≥ `fastpath_block` | precision (were these actually fraud?) |
| Agents | ambiguous zone | precision, recall, F1 |

**What to do:**
- In `helpers/batch_runner.py`, tag each result with its decision path (`hard_rule`, `fastpath_approve`, `fastpath_block`, `agent`)
- In `app.py` batch tab, add a "Per-layer breakdown" section below the existing metrics

### 2b. LLM-as-judge reasoning evaluator

Use Claude Sonnet to score agent reasoning quality on 4 dimensions (1–5 each):
- **consistency** — does the decision follow logically from the evidence?
- **specificity** — are fraud signals concrete (e.g. "3 txns in 60s") vs vague ("unusual activity")?
- **calibration** — is the confidence appropriate given the evidence?
- **explanation_quality** — is the customer explanation clear and accurate?

**What to do:**
- Add `helpers/agent_evaluator.py` with a `judge_agent_reasoning(assessment, fraud_score, actual_label)` function
- Call it on a sample of agent-handled transactions in batch mode (opt-in, adds cost)
- Show average scores per dimension in the batch tab agent section

### 2c. Agent calibration plot

Check whether agent confidence scores are meaningful — plot agent confidence (binned) vs actual accuracy. Want the diagonal; overconfident agents will be above it.

**What to do:**
- In batch results, bin `confidence` into 10 buckets
- For each bucket, compute actual accuracy (`decision in {BLOCK, REVIEW}` == `actual == 1`)
- Add a calibration scatter plot in the batch tab agent section

### 2d. Override rate tracking

When `threshold_overrode_agent` is True, that decision is logged. Surface this in the analytics tab:
- What % of agent-handled transactions get overridden by the threshold?
- Of the overridden cases, what's the actual fraud rate? (If high, threshold is right to override; if low, agents were right)

**What to do:**
- Add `threshold_override` boolean to the session history entry
- Show override rate metric in Tab 3 analytics
- Add a breakdown: override cases vs non-override cases, by actual label
