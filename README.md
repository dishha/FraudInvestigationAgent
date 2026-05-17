# Fraud Investigation Engine

A production-grade fraud detection system combining a LightGBM scoring model with a 4-agent Claude AI reasoning pipeline. Each transaction is scored by ML, investigated by specialized agents, and routed to a final decision — all surfaced through an interactive Streamlit dashboard.

---

## Architecture

```
Transaction
    │
    ▼
┌─────────────────────────────────────────────────┐
│              LightGBM Scorer                    │
│  460 features · 843 trees · AUC 0.898           │
│  Isotonic calibration → fraud probability       │
└────────────────────┬────────────────────────────┘
                     │ fraud score
                     ▼
┌─────────────────────────────────────────────────┐
│         Adaptive Tool Selector                  │
│  Score-gated: skips tools when ML is confident  │
│  ┌────────────┐ ┌─────────────┐ ┌────────────┐  │
│  │Card Velocity│ │Addr Cluster │ │Email Domain│  │
│  │  Checker   │ │  Checker    │ │  Checker   │  │
│  └────────────┘ └─────────────┘ └────────────┘  │
└────────────────────┬────────────────────────────┘
                     │ tool results
                     ▼
┌─────────────────────────────────────────────────┐
│           4-Agent Reasoning Pipeline            │
│                                                 │
│  MLAnalyst → Investigator → RiskAssessor        │
│                                  → Explainer    │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
         APPROVE / SOFT_DECLINE / REVIEW / BLOCK
```

### Decision Thresholds

| Decision | Score Range | Meaning |
|---|---|---|
| `APPROVE` | < 0.50 | Low risk, proceed |
| `SOFT_DECLINE` | 0.50 – 0.65 | Borderline, soft block |
| `REVIEW` | 0.65 – 0.85 | Elevated risk, queue for review |
| `BLOCK` | ≥ 0.85 | High confidence fraud, hard block |

---

## Model Performance

Trained on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (590k transactions, 2017–2018 e-commerce).

| Metric | Value |
|---|---|
| AUC-ROC | 0.898 |
| PR-AUC | 0.536 |
| Brier Score | 0.0223 |
| Precision @ 0.5 | 0.787 |
| Recall @ 0.5 | 0.356 |
| Training time | 29.5s |
| Features | 460 |
| Trees | 843 |

**Temporal split** — no future data leaks into training:

| Split | Rows | Fraud Rate |
|---|---|---|
| Train | 413,713 | 3.54% |
| Val | 83,894 | 3.40% |
| Test | 87,292 | 3.48% |

---

## Project Structure

```
fraud-investigation-engine/
├── app.py                          # Streamlit UI (5 tabs)
├── config.py                       # Thresholds, colors, log paths
├── requirements.txt
├── dockerfile
│
├── scripts/
│   ├── feature_engineering.py      # Offline feature pipeline + inference helper
│   ├── ml_pipeline.py              # Offline training pipeline + score_with_ml()
│   └── fraud_investigation_agents.py  # Agent orchestration entry points
│
├── agents/
│   ├── investigator.py             # FraudInvestigationOrchestrator (Claude API)
│   ├── tool_execution.py           # ToolExecutor — runs investigation tools
│   ├── tool_selection.py           # AdaptiveToolSelector — score-gated dispatch
│   ├── dialogue.py                 # Inter-agent dialogue helpers
│   └── learning.py                 # Feedback and learning utilities
│
├── helpers/
│   ├── loaders.py                  # Cached @st.cache_resource loaders
│   ├── batch_runner.py             # ML batch scoring + agent batch pipeline
│   ├── decision_strategy.py        # Decision merging, normalization, fallbacks
│   └── feature_attribution.py     # SHAP helpers
│
├── tools/
│   └── queries.py                  # CardVelocityChecker, AddressClusterChecker,
│                                   # EmailDomainChecker (all temporally gated)
│
├── notebooks/
│   ├── data_prep.ipynb             # IEEE-CIS EDA and temporal split
│   ├── feature_engineering.ipynb   # Feature exploration
│   └── model_training.ipynb        # Training experiments
│
├── model/
│   ├── lightgbm_model.pkl          # Trained model
│   ├── calibrator.pkl              # Isotonic regression calibrator
│   ├── feature_names.pkl           # Ordered feature list
│   ├── feature_importance.csv      # Gain-based importance
│   ├── metrics.json                # AUC, PR-AUC, Brier (loaded by UI)
│   ├── MODEL_CARD.md               # Full model documentation
│   └── results/                    # ROC, PR, SHAP, importance plots
│
├── data/                           # Not committed — see Data Setup below
│   ├── ieee_prepared.pkl
│   ├── features_train.pkl
│   ├── features_val.pkl
│   └── features_test.pkl
│
├── tests/
│   └── test_investigator_agents.py # 25 unit tests
│
└── logs/
    └── fraud_detection.log         # Append-only structured log
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data setup

Download the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset from Kaggle and place the CSVs in `data/raw/`. Then run the preparation notebooks in order:

```bash
# From project root
jupyter notebook notebooks/data_prep.ipynb          # produces data/ieee_prepared.pkl
python scripts/feature_engineering.py               # produces data/features_*.pkl
```

> Both scripts must be run from inside the `scripts/` directory, or adapt paths as needed.

### 3. Train the model

```bash
cd scripts
python ml_pipeline.py
```

Outputs to `model/`: `lightgbm_model.pkl`, `calibrator.pkl`, `feature_names.pkl`, `feature_importance.csv`, `metrics.json`, `MODEL_CARD.md`.

### 4. Configure the Anthropic API key (optional)

The agent pipeline uses Claude. Without a key the system runs in **mock mode** (simulated agent responses — fully functional for development).

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Docker

```bash
docker build -t fraud-engine .
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  fraud-engine
```

Open [http://localhost:8501](http://localhost:8501).

---

## UI Tabs

| Tab | Description |
|---|---|
| **Score Transaction** | Pick any test transaction by index, run the full pipeline, see the decision, SHAP explanation, and agent trace |
| **Batch Evaluation** | Score up to 2,000 transactions, view AUC/PR curves, confusion matrix, threshold sweep, and optional agent metrics |
| **Analytics** | Session-level dashboard — decision distribution, score trends, accuracy over scored history |
| **Logs** | Live log viewer with level filtering and full-log download |
| **Settings** | System info, model stats, file status, feature importance chart |

---

## Agent Pipeline

Four specialized Claude agents run in sequence for each transaction:

| Agent | Role |
|---|---|
| **ML Analyst** | Interprets the fraud score and top SHAP features |
| **Transaction Investigator** | Queries tool results — card velocity, address clusters, email domain history |
| **Risk Assessor** | Synthesizes ML + investigation evidence into a risk level (LOW / MEDIUM / HIGH / CRITICAL) |
| **Decision Explainer** | Issues a final decision and a customer-facing explanation |

**Tool selection** — in `adaptive` mode the selector skips all tools when the ML score is already high-confidence (very high or very low), saving latency. In `all` mode every tool always runs.

**Mock mode** — set `MOCK_MODE = True` in `agents/investigator.py` or omit the API key to run without Claude API calls.

---

## Investigation Tools

All tools query only historical data (train + val splits). The test set is never passed as context, preventing agent-layer data leakage.

| Tool | What it checks |
|---|---|
| `CardVelocityChecker` | Transaction frequency and amount variance for the same card in recent history |
| `AddressClusterChecker` | Number of distinct cards seen at the same billing address |
| `EmailDomainChecker` | Historical fraud rate and average amount for the purchaser's email domain |

Each tool applies a **temporal gate** — only transactions with `TransactionDT < current_txn_time` are queried, preventing future-data leakage within the historical window.

---

## Feature Engineering

Key design decisions (zero cross-split leakage):

- **Outlier removal** — 99th-percentile cap computed from train only, applied to all splits
- **Column pruning** — high-missing (>99%) and zero-variance columns detected from train only
- **Temporal features** — hour, day-of-week, cyclic sin/cos, is_night
- **Amount features** — log transform, squared, small/medium/large buckets
- **D-column normalization** — days-since values: `transaction_day − D_col`
- **Group statistics** — card, device, email, address, hour stats computed from train only and mapped to all splits
- **Categorical encoding** — pandas `Categorical` codes; categories defined from train; unseen → `-1` (LightGBM handles natively)

> **Known limitation**: within-train target-encoded features (`device_fraud_rate`, `email_fraud_rate`, `address_fraud_rate`, `hour_fraud_rate`) include each training row's own label in its group mean. Out-of-fold encoding would eliminate this minor within-train leakage.

---

## Running Tests

```bash
pytest tests/ -v
```

25 unit tests covering the agent pipeline, tool execution, and decision logic.

---

## Configuration

All thresholds and paths are in `config.py`:

```python
BLOCK_THRESHOLD        = 0.85
REVIEW_THRESHOLD       = 0.65
SOFT_DECLINE_THRESHOLD = 0.50
```

---

## Requirements

- Python 3.9+
- LightGBM 4.5+
- Streamlit 1.37+
- Anthropic SDK 0.34+ (optional — mock mode available)
- See `requirements.txt` for full pinned versions
