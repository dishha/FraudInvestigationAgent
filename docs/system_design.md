# Fraud Investigation Engine — System Design

---

## 1. Problem Statement

Real-time fraud detection for card-not-present (CNP) e-commerce transactions requires:

- **Sub-second decisions** on individual transactions (P95 < 500ms for ML path, < 3s for agentic path)
- **High precision** — false positives cost legitimate customer revenue
- **High recall** — false negatives cost direct fraud losses
- **Explainability** — regulators and customers require human-readable reasoning
- **Graceful degradation** — system must function if the LLM API is unavailable

---

## 2. High-Level Architecture

The system is a **layered decision pipeline** that progressively applies more expensive (but more accurate) reasoning, short-circuiting as soon as confidence is sufficient.

```
┌────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI  (app.py)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────────┐  │
│  │Score Txn tab │  │Batch Eval tab│  │Analytics  │  │Settings/Logs │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┘  └──────────────┘  │
└─────────┼────────────────┼────────────────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DECISION PIPELINE                                │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  LAYER 0 — Hard Rules Engine           ⚡ <1ms, deterministic │  │
│  │  • Blacklisted device                                        │  │
│  │  • Card velocity ≥3/min or ≥10/hr                           │  │
│  │  • Amount >$500 on first-seen device                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│            │ (no rule triggered)                                    │
│            ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  LAYER 1 — ML Scoring                  ⚡ <100ms             │  │
│  │  • LightGBM (843 trees, 460 features)                        │  │
│  │  • Isotonic calibration → score [0, 1]                       │  │
│  │  • Fast-path: score≥0.72 → BLOCK, score<0.01 → APPROVE      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│            │ (score 0.01–0.72, uncertain zone)                     │
│            ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  LAYER 2 — Agentic Investigation       🤖 ~500ms–2s          │  │
│  │  ┌────────────────────────┐  ┌───────────────────────────┐   │  │
│  │  │  Investigation Tools   │  │   4-Agent Chain (Claude)  │   │  │
│  │  │  • CardVelocityChecker │  │   1. MLAnalyst            │   │  │
│  │  │  • DeviceProfileChecker│  │   2. TransactionInvestiga │   │  │
│  │  │  • EmailDomainChecker  │  │   3. RiskAssessor         │   │  │
│  │  │  • AddressClusterChkr  │  │   4. DecisionExplainer    │   │  │
│  │  └────────────────────────┘  └───────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│            │                                                        │
│            ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  LAYER 3 — Decision Governance                               │  │
│  │  • merge(agent_decision, threshold_decision) → max risk      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
   Final Decision + Explanation + SHAP values + Cost + Latency
```

---

## 3. Component Specifications

### 3.1 Layer 0 — Hard Rules Engine (`helpers/rules_engine.py`)

| Attribute | Value |
|---|---|
| **Class** | `HardRulesEngine` |
| **Latency** | <1ms (pure Python, no I/O) |
| **Rules** | 3 (blacklist, velocity, large-amount-new-device) |
| **Output** | `RuleResult(triggered, decision, rule_name, evidence)` |
| **Override** | No — rule decisions are final |
| **Rationale** | Certain fraud patterns require zero-tolerance, zero-latency response independent of ML confidence |

**Rules in priority order:**

1. `_rule_blacklisted_device()` — device_id on static blocklist → `BLOCK`
2. `_rule_card_velocity()` — ≥3 txns/60s OR ≥10 txns/3600s on same card → `BLOCK`
3. `_rule_large_amount_new_device()` — amount > $500 AND device never seen before → `BLOCK`

---

### 3.2 Layer 1 — ML Scoring (`scripts/ml_pipeline.py`)

| Attribute | Value |
|---|---|
| **Model** | LightGBM binary classifier |
| **Trees** | 843 (early stopped from 1000-round budget) |
| **Features** | 460 engineered features |
| **Calibration** | Isotonic regression fitted on validation set |
| **AUC-ROC** | 0.898 |
| **PR-AUC** | 0.536 |
| **Brier Score** | 0.0223 |
| **Training data** | 413,713 transactions (IEEE-CIS 2017–2018) |
| **Inference latency** | <100ms |

**Feature categories (460 total):**

| Category | Examples | Count |
|---|---|---|
| Temporal | hour, day_of_week, hour_sin/cos | ~10 |
| Amount | log_amount, amount_sq, amount_bucket | ~5 |
| Transaction | ProductCD, card type, addr encodings | ~50 |
| D-columns | days-since features (normalized) | ~15 |
| V-columns | Vesta anonymized signals | ~339 |
| Group stats | card_fraud_rate, device_txn_count, email_domain_freq | ~41 |

**Fast-path thresholds (from `thresholds.json`):**

```
score < 0.01  → APPROVE  (skip agents, ~85% of legitimate traffic)
score ≥ 0.72  → BLOCK    (skip agents, ~60% of fraud traffic)
0.01–0.72     → agentic  (~15% of traffic requiring deeper analysis)
```

---

### 3.3 Layer 2 — Agentic Investigation

#### 3.3.1 Adaptive Tool Selector (`agents/tool_selection.py`)

Score-gated dispatcher with 2000ms latency budget:

| Score Range | Tools Invoked |
|---|---|
| < 0.2 or > 0.8 | None (skip tools) |
| 0.2–0.8 | velocity → device → email → address (budget permitting) |

Tools run in priority order: velocity first (most predictive), address last.

#### 3.3.2 Investigation Tools (`tools/queries.py`)

All tools query **`hist_raw` only (train + val set)** — no test-set leakage.

| Tool | Signals Detected | Key Thresholds |
|---|---|---|
| `CardVelocityChecker` | Card testing, rapid spending, multi-country | 3+ in 1hr (<$50), 5+ in 1hr, 2+ countries in 24hr |
| `DeviceProfileChecker` | New device, device fraud rate, multi-card | fraud_rate > 10%, num_cards > 3, age < 1hr |
| `EmailDomainChecker` | Temp email, high domain fraud rate | Mailinator etc. → instant flag, fraud_rate > 8% |
| `AddressClusterChecker` | Fraud ring, high-volume cluster | 5+ cards + 12%+ fraud → fraud ring |

#### 3.3.3 Four-Agent Chain (`agents/investigator.py`)

Sequential chain where each agent receives the previous agent's output:

```
Transaction + ML Score + Tool Results
         │
         ▼
   ┌─────────────┐     Prompt: "Interpret this fraud score and top SHAP features"
   │  MLAnalyst  │ ──► Output: {fraud_score, confidence, top_features, caveats}
   └─────────────┘
         │
         ▼
   ┌──────────────────────────┐   Prompt: "Analyze these tool signals for fraud patterns"
   │ TransactionInvestigator  │ ──► Output: {fraud_signals[], overall_risk, summary}
   └──────────────────────────┘
         │
         ▼
   ┌──────────────┐   Prompt: "Synthesize ML + signals into a risk judgment"
   │ RiskAssessor │ ──► Output: {risk_level, risk_score, confidence, recommendation}
   └──────────────┘
         │
         ▼
   ┌──────────────────┐   Prompt: "Write a customer-friendly explanation"
   │ DecisionExplainer│ ──► Output: plain-text explanation
   └──────────────────┘
```

**LLM configuration:**
- Model: `claude-haiku-4-5-20251001`
- Cost: $0.80/1M input tokens, $4.00/1M output tokens
- Fallback: `MOCK_MODE=True` returns deterministic hardcoded responses

---

### 3.4 Layer 3 — Decision Governance (`helpers/decision_strategy.py`)

**Decision risk ordering:** APPROVE (0) < SOFT_DECLINE (1) < REVIEW (2) < BLOCK (3)

**Merge strategy:** `final_decision = max_risk(agent_decision, threshold_decision)`

- Agent recommendation can **escalate** the threshold decision (e.g., threshold says REVIEW but agent says BLOCK → BLOCK)
- Agent recommendation **cannot de-escalate** a threshold decision (conservative governance)
- If agent fails or returns invalid output → threshold decision used as fallback

**Threshold decision mapping:**

| Score | Decision |
|---|---|
| < 0.07 | APPROVE |
| 0.07–0.21 | SOFT_DECLINE |
| 0.21–0.53 | REVIEW |
| ≥ 0.53 | BLOCK |

---

## 4. Data Model

### Transaction Input (raw fields used by tools)

```python
{
    "TransactionID": str,
    "TransactionDT": int,        # seconds since reference date
    "TransactionAmt": float,
    "card1": int,                # card identifier
    "card2–6": various,          # card metadata
    "addr1", "addr2": float,     # billing address codes
    "P_emaildomain": str,        # purchaser email domain
    "DeviceType": str,
    "DeviceInfo": str,
    "id_30": str,                # OS info
    "id_31": str,                # browser info
    # ... 460 total engineered features for ML
}
```

### FraudAssessment Output

```python
@dataclass
class FraudAssessment:
    transaction_id: str
    final_decision: str          # APPROVE / SOFT_DECLINE / REVIEW / BLOCK
    risk_level: str              # LOW / MEDIUM / HIGH / CRITICAL
    risk_score: float            # 0.0–1.0
    confidence: str              # LOW / MEDIUM / HIGH
    fraud_score: float           # calibrated ML score
    ml_analysis: dict            # MLAnalyst output
    investigation: dict          # TransactionInvestigator output
    risk_assessment: dict        # RiskAssessor output
    customer_explanation: str    # DecisionExplainer output
    tool_results: dict           # raw tool outputs
    execution_time_ms: float
    total_cost_usd: float        # sum of all agent API costs
    agents_invoked: list[str]
```

### ToolResult

```python
@dataclass
class ToolResult:
    result: dict                 # tool-specific payload
    confidence: float            # 0.0–1.0
    data_points: int             # rows queried from hist_raw
    error: Optional[str]
    execution_time_ms: float
```

---

## 5. Training Pipeline

```
Raw Data (IEEE-CIS Kaggle)
        │
        ▼
data_prep.ipynb — Temporal split (no random shuffle)
        │
        ├── Train  (Jan–Sep 2017, 413k rows)
        ├── Val    (Oct–Nov 2017,  84k rows)
        └── Test   (Dec 2017–2018, 87k rows)
                │
                ▼
feature_engineering.py — 460 features
        │ Fit on TRAIN only:
        │  • Group stats (card, device, email, addr fraud rates)
        │  • Outlier caps (99th percentile)
        │  • Categorical maps (unseen → -1)
        │  • Temporal features (sin/cos of hour/DOW)
        │
        ▼
ml_pipeline.py — Training
        │
        ├── LightGBM.train(train, val, early_stopping=50/1000)
        ├── IsotonicRegression.fit(val_predictions, val_labels)
        ├── threshold_optimizer.fit(val) → thresholds.json
        ├── SHAP.TreeExplainer(model) → feature_importance.csv
        └── Metrics: AUC=0.898, PR-AUC=0.536, Brier=0.0223
                │
                ▼
        model/ artifacts (all versioned together)
```

**Data leakage controls:**
- Temporal split (not random) — no future data in training
- Group statistics derived from train only, applied to val/test
- Threshold optimization on val (not test)
- `hist_raw` for tool queries = train + val only (test is the hold-out)

---

## 6. Scalability & Production Considerations

### Current State (MVP / Research)

| Dimension | Current | Notes |
|---|---|---|
| **Serving** | Streamlit (single process) | Development only |
| **Throughput** | ~1 txn/sec (agent path) | Claude API rate limit bound |
| **ML latency** | <100ms | Synchronous, in-process |
| **Agent latency** | 500ms–3s | 4 sequential API calls |
| **Batch** | Sequential Python loop | No parallelism |
| **State** | Streamlit session state | Ephemeral, no DB |
| **Auth** | None | Open dev server |

### Production Path

```
Current                        →    Production Target
─────────────────────────────────────────────────────
Streamlit app                  →    FastAPI + React dashboard
Single process                 →    Kubernetes pods (autoscaling)
In-memory model loading        →    Model serving (TorchServe / BentoML)
Sequential agent calls         →    Async parallel (asyncio + Claude Batch API)
No queue                       →    Kafka / SQS for transaction ingestion
Streamlit session state        →    PostgreSQL + Redis cache
Manual model artifacts         →    MLflow model registry + CI/CD retraining
MOCK_MODE for testing          →    Feature flags + circuit breakers
No auth                        →    OAuth2 / mTLS for internal services
```

### Latency Budget (production target)

```
P50: 80ms   (85% fast-path ML)
P95: 400ms  (fast-path ML + occasional tool call)
P99: 2500ms (full 4-agent pipeline)
```

---

## 7. Known Limitations & Technical Debt

| Issue | Severity | Notes |
|---|---|---|
| **Within-train target leakage** | Medium | Group stats (e.g., `device_fraud_rate`) include each row's own label. Fix: out-of-fold group statistics. Model still strong (AUC 0.898) but will overfit train split. |
| **IEEE-CIS data staleness** | Medium | Model trained on 2017–2018 data. Fraud patterns shift; production model needs regular retraining on live data. |
| **Agent cost not surfaced in UI** | Low | `FraudAssessment.total_cost_usd` is computed but not displayed in the Streamlit UI (see NEXT_STEPS.md). |
| **Sequential agent calls** | Low | 4 Claude API calls are serial. Parallelizable after tool execution with Claude Batch API or asyncio. |
| **No persistent history** | Low | Transaction history lives in Streamlit session state; cleared on browser refresh. Needs a database for production use. |
| **MOCK_MODE hardcoded responses** | Low | Mock responses always return the same values, preventing realistic agent testing without API keys. |
| **No drift detection** | Medium | No monitoring for data drift or model degradation in production. |

---

## 8. Security & Privacy

| Control | Status |
|---|---|
| API key storage | `.env` file (gitignored) |
| PII in agent prompts | Transaction metadata (card IDs, email domains) sent to Claude API — evaluate data residency requirements |
| Test data exposure | IEEE-CIS data is public Kaggle dataset; no real cardholder data |
| Input validation | Minimal — tool queries rely on pandas operations, not raw SQL |
| Auth | None (development mode) |

**For production:**
- Tokenize/hash card IDs and email addresses before sending to LLM
- Implement request signing for internal service calls
- Add rate limiting on the API layer
- Audit log all BLOCK decisions (regulatory requirement in most jurisdictions)

---

## 9. Cost Model

### Per-transaction cost (agent path only, ~15% of traffic)

| Agent | Approx Input Tokens | Approx Output Tokens | Cost @ Haiku-4-5 rates |
|---|---|---|---|
| MLAnalyst | ~800 | ~200 | $0.00144 |
| TransactionInvestigator | ~1200 | ~300 | $0.00216 |
| RiskAssessor | ~1500 | ~200 | $0.00200 |
| DecisionExplainer | ~1000 | ~150 | $0.00140 |
| **Total** | ~4500 | ~850 | **~$0.007/txn** |

At 1M transactions/day with 15% hitting the agent path:
- Agent path: 150,000 txns × $0.007 = **$1,050/day**
- Optimize by: (a) tighter fast-path thresholds, (b) Claude Batch API (50% discount), (c) caching similar cases

---

## 10. Monitoring & Observability

### Current

- `logs/fraud_detection.log` — append-only structured log (visible in Settings tab)
- `FraudAssessment.execution_time_ms` — per-transaction latency
- `FraudAssessment.total_cost_usd` — per-transaction agent cost
- Streamlit session history — in-memory analytics tab

### Production Additions Needed

| Signal | Tooling |
|---|---|
| Model score distribution | Prometheus + Grafana |
| Decision distribution (BLOCK %, REVIEW %) | Same |
| Agent latency P50/P95/P99 | Same |
| Claude API error rate | Same |
| Data drift (PSI on key features) | EvidentlyAI or custom |
| Fraud rate feedback loop | Label store → retraining trigger |
| LangSmith traces | Already configured in `.env` — wire up |
