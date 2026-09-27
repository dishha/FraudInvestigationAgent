# Fraud Investigation Engine — Pipeline Walkthrough

End-to-end technical walkthrough: raw data → feature engineering → model training → inference → Streamlit UI.

| Version | Split strategy | Script |
|---|---|---|
| **v1** (current) | Temporal — first 67% / 10% / 23% by `TransactionDT` | `data_prep.ipynb` |
| **v2** | Card-based — all txns for a card go to one split | `scripts/data_split_v2.py` |

Run v2 pipeline:
```bash
python scripts/data_split_v2.py                          # → data/ieee_prepared_v2.pkl
python scripts/feature_engineering.py --split v2         # → data/features_{train,val,test}_v2.pkl
python scripts/ml_pipeline.py                            # point IN_FILE vars at _v2 pickles
```

---

## 1. Data Preparation

**Dataset:** IEEE-CIS Fraud Detection (Kaggle, 2017–2018). Two CSV files joined on `TransactionID`:
- `train_transaction.csv` — 590k rows, 394 columns (amounts, card metadata, Vesta V-features)
- `train_identity.csv` — 144k rows, 41 columns (device info, browser, OS, network)

**Split strategy: temporal, not random.**

Random splits leak future information into training because fraud patterns are time-correlated. The split is done by `TransactionDT` (seconds since a reference epoch):

| Split | Period | Rows | Fraud rate |
|---|---|---|---|
| Train | Jan–Sep 2017 | ~413,713 | ~3.5% |
| Val | Oct–Nov 2017 | ~84,000 | ~3.5% |
| Test | Dec 2017–2018 | ~87,000 | ~3.5% |

**Why val set matters:** The val set is used for three separate things:
1. Early stopping signal during LightGBM training
2. Fitting the Isotonic calibrator
3. Deriving decision thresholds

This means val set labels influence both the model and the thresholds — the test set is the only truly held-out evaluation.

**Outputs:** `data/ieee_prepared.pkl` (raw splits), `data/features_{train,val,test}.pkl` (engineered).

---

## 2. Feature Engineering (`scripts/feature_engineering.py`)

460 features total, built in categories:

### 2.1 Temporal features (~10)
- `hour`, `day_of_week`, `day_of_month` extracted from `TransactionDT`
- `hour_sin`, `hour_cos` — cyclical encoding so hour 23 and hour 0 are close
- `is_weekend`, `is_night` (22:00–06:00) — binary indicators

### 2.2 Amount features (~5)
- `log_amount` — log1p transform to reduce skew
- `amount_sq` — quadratic term for large-amount sensitivity
- `amount_bucket` — ordinal binning (<$10, $10–50, $50–200, $200–1k, >$1k)

### 2.3 Transaction metadata (~50)
- `ProductCD`, card type, issuer bank, billing country — label-encoded
- `addr1`, `addr2` — billing zip codes, encoded with frequency
- `P_emaildomain`, `R_emaildomain` — purchaser and recipient email domains

### 2.4 D-columns (~15)
Days-since features (D1–D15). Represent time gaps (e.g., days since last transaction on card). Normalized by subtracting the within-card mean from training data.

### 2.5 V-columns (~339)
Vesta proprietary anonymized signals (V1–V339). Sparse — many are NaN. Filled with column median from training data; missing indicator columns added where >5% null.

### 2.6 Group statistics (~41) — most predictive
Derived from training data only, applied to all splits:

| Feature | Computation | Signal |
|---|---|---|
| `card_fraud_rate` | mean(isFraud) grouped by card1 in train | Cards with fraud history |
| `device_fraud_rate` | mean(isFraud) grouped by DeviceInfo in train | Risky devices |
| `email_domain_fraud_rate` | mean(isFraud) grouped by P_emaildomain in train | Risky email providers |
| `device_txn_count` | count grouped by DeviceInfo in train | New vs seen device |
| `addr_fraud_rate` | mean(isFraud) grouped by addr1 in train | Risky billing regions |
| `card_amt_mean`, `card_amt_std` | amount stats per card in train | Spending baseline |

**Leakage note:** Within the training set, each row's own label is included in its group statistic. This produces a small upward bias on training AUC. The model card documents this. Fix for production: out-of-fold group statistics (5-fold CV on train only).

### 2.7 Transform at inference
`transform_features_for_scoring(df, feature_names)` applies the same transforms using the fitted objects saved in `model/feature_names.pkl`. New/unseen categories map to `-1`. Column order is enforced to match training.

---

## 3. Model Training (`scripts/ml_pipeline.py`)

### 3.1 LightGBM configuration

```python
params = {
    'objective':        'binary',
    'metric':           'auc',
    'learning_rate':    0.05,
    'num_leaves':       31,        # max leaf nodes per tree
    'max_depth':        8,         # cap tree depth
    'feature_fraction': 0.8,       # 80% features per tree (prevents correlation)
    'bagging_fraction': 0.8,       # 80% rows per tree (reduces overfitting)
    'bagging_freq':     5,         # re-sample every 5 rounds
    'min_data_in_leaf': 20,        # minimum samples per leaf
    'lambda_l1':        0.1,       # L1 regularization
    'lambda_l2':        0.1,       # L2 regularization
    'seed':             42,
}
```

Training runs up to 1000 boosting rounds with `early_stopping(50)` — stops if val AUC doesn't improve for 50 consecutive rounds. Final model: **843 trees**.

### 3.2 Calibration

LightGBM outputs a raw score in [0, 1], but it is not a calibrated probability — the score distribution is concentrated near 0 and 1 rather than matching the actual fraud rate (~3.5%).

**Isotonic regression** (`sklearn.isotonic.IsotonicRegression`) fits a monotone step function mapping raw scores → calibrated probabilities using val set predictions and labels:

```python
val_pred_raw   = model.predict(X_val, num_iteration=model.best_iteration)
calibrator     = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(val_pred_raw, y_val)
```

After calibration: Brier score improves from ~0.029 → 0.0223 (~23% reduction). The calibrated score is used everywhere downstream.

### 3.3 Threshold derivation

Thresholds are derived on the **val set** (not test) by sweeping 5000 candidate values from 0.005 to 0.995 and finding the minimum score where precision hits a target:

| Threshold name | Target precision | Derived score | Action |
|---|---|---|---|
| `fastpath_approve` | < 10% | ~0.01 | Skip agents → APPROVE |
| `soft_decline` | ≥ 30% | ~0.07 | SOFT_DECLINE (3DS challenge) |
| `review` | ≥ 60% | ~0.21 | Human REVIEW queue |
| `block` | ≥ 80% | ~0.53 | Hard BLOCK |
| `fastpath_block` | ≥ 90% | ~0.72 | Skip agents → BLOCK |

Saved to `model/thresholds.json`. Also saved: per-zone fraud rates (observed on val) for the 6 decision zones.

### 3.4 SHAP explanations

`shap.TreeExplainer(model)` computes exact SHAP values via the TreeSHAP algorithm — O(TLD) where T=trees, L=leaves, D=depth. At inference, SHAP values for a single transaction explain which features pushed the score up or down from the expected value (base rate).

### 3.5 Artifacts saved

```
model/
├── lightgbm_model.pkl       # full model object (via pickle)
├── lightgbm_model.txt       # LightGBM native format (portable)
├── calibrator.pkl           # IsotonicRegression object
├── feature_names.pkl        # ordered list of 460 feature names
├── feature_importance.csv   # gain-based importance, all features
├── thresholds.json          # 5 thresholds + zone fraud rates
├── MODEL_CARD.md            # metrics, limitations, intended use
└── results/
    ├── feature_importance.png
    ├── shap_summary.png
    ├── evaluation_curves.png   # ROC + PR
    └── threshold_analysis.png  # precision/recall vs threshold + zone bar chart
```

### 3.6 Test set metrics

| Metric | Value |
|---|---|
| AUC-ROC | 0.898 |
| PR-AUC | 0.536 |
| Brier Score | 0.0223 |
| Precision @ 0.5 threshold | ~0.74 |
| Recall @ 0.5 threshold | ~0.47 |

---

## 4. Inference — The 4-Layer Decision Pipeline

Every call to `score_with_ml` + agent pipeline in `app.py` runs through four layers in order. Each layer can short-circuit and return a final decision, bypassing all subsequent layers.

### Layer 0 — Hard Rules (`helpers/rules_engine.py`)

Three deterministic rules evaluated in priority order:

```
1. BLACKLISTED_DEVICE   — DeviceInfo in static blocklist         → BLOCK
2. CARD_VELOCITY_1M     — card1: ≥3 transactions in last 60s     → BLOCK
3. CARD_VELOCITY_1H     — card1: ≥10 transactions in last 3600s  → BLOCK
4. LARGE_AMOUNT_NEW_DEVICE — amount > $500 AND device never seen  → BLOCK
```

`hist_raw` (train + val transactions) is queried to count prior transactions for the same card/device. Latency: <1ms (pure pandas boolean indexing, no I/O).

If any rule fires: `RuleResult(triggered=True, rule_name=..., evidence=...)`. ML scoring and agents are skipped entirely.

### Layer 1 — ML Scoring (`scripts/ml_pipeline.py → score_with_ml`)

```python
def score_with_ml(transaction_features, models):
    aligned       = transaction_features[feature_names]   # enforce column order
    raw_score     = float(model.predict(aligned)[0])      # LightGBM leaf sum → sigmoid
    calib_score   = float(calibrator.predict([raw_score])[0])  # Isotonic mapping
    importance     = model.feature_importance(importance_type='gain')
    shap_values   = shap.TreeExplainer(model).shap_values(aligned)
    return {'fraud_score': calib_score, 'fraud_score_raw': raw_score,
            'shap_values': shap_values, 'feature_importance': top_10}
```

**Fast-path gating (from `config.py`):**

| Condition | Path | Agents? |
|---|---|---|
| `score ≥ ML_FASTPATH_BLOCK` (0.72) | → BLOCK immediately | No |
| `score < ML_FASTPATH_APPROVE` (0.01) AND `tool_mode != "all"` | → APPROVE immediately | No |
| Otherwise (0.01–0.72) | → proceed to Layer 2 | Yes |

The fast-path covers ~85% of legitimate traffic (approve) and ~60% of fraud (block), meaning agents only run on the ~15% of transactions that are genuinely ambiguous.

### Layer 2 — Agentic Investigation

#### Tool execution (`scripts/fraud_investigation_agents.py → investigate_transaction`)

Before agents run, 4 investigation tools query `hist_raw`:

| Tool | What it computes | Key output |
|---|---|---|
| `CardVelocityChecker` | Transactions by same card in last 1h, 6h, 24h; amount stats; countries | `txn_1h`, `countries_24h`, `card_testing_signal` |
| `DeviceProfileChecker` | First-seen time, fraud rate on this device, number of distinct cards | `device_age_hrs`, `device_fraud_rate`, `num_cards_on_device` |
| `EmailDomainChecker` | Domain type (temp/free/corporate), fraud rate for this domain | `is_temp_email`, `domain_fraud_rate` |
| `AddressClusterChecker` | Number of cards sharing this billing address, cluster fraud rate | `cluster_size`, `cluster_fraud_rate`, `fraud_ring_signal` |

In **adaptive mode**: tools are skipped if score < 0.2 or > 0.8 (signal is already strong enough). In **all tools mode**: all 4 tools always run.

#### 4-Agent chain (`agents/investigator.py → FraudInvestigationOrchestrator`)

Sequential chain — each agent's output becomes context for the next:

**Agent 1 — MLAnalyst**
- Input: calibrated score, raw score, top-10 feature importances
- Prompt prefilled with `{` to force JSON output
- Output: `{fraud_score, calibrated_probability, confidence, top_features[], summary, caveats}`
- Validates with `validate_ml_analysis()` — normalizes types, clips floats to [0,1]

**Agent 2 — TransactionInvestigator**
- Input: serialized tool results dict (card velocity, device profile, email, address)
- Detects 4 patterns: card testing, account takeover, location anomalies, device anomalies
- Output: `{fraud_signals[], contradicting_signals[], summary, overall_risk}`

**Agent 3 — RiskAssessor**
- Input: MLAnalyst output + TransactionInvestigator output
- Applies risk framework: CRITICAL (>0.85) / HIGH (0.65–0.85) / MEDIUM (0.40–0.65) / LOW (<0.40)
- Output: `{risk_level, risk_score, confidence, key_signals[], recommendation, reasoning}`
- Validates with `validate_risk_assessment()` — normalizes `risk_level` and `recommendation` strings

**Agent 4 — DecisionExplainer**
- Input: decision, risk level, key signals, transaction amount
- Output: customer-facing plain English explanation (not JSON)
- Template-guided: different templates for BLOCK / SOFT_DECLINE / REVIEW / APPROVE

**All agents use:**
- Model: `claude-haiku-4-5-20251001`
- Temperature: 0 (deterministic)
- Max tokens: 500–1500 depending on agent
- Prefill `{"` trick: assistant turn starts with `{` to steer output toward JSON

**MOCK_MODE** (`config.MOCK_MODE = True`): returns hardcoded deterministic responses for each agent. Used when `ANTHROPIC_API_KEY` is not set.

**Cost tracking:** each agent stores `last_usage = {input_tokens, output_tokens}`. After all 4 run, the orchestrator sums and computes:

```python
agent_cost_usd = (total_input * CLAUDE_INPUT_COST_PER_TOKEN
                + total_output * CLAUDE_OUTPUT_COST_PER_TOKEN)
```

Rates from `config.py`: $0.80/1M input, $4.00/1M output (Haiku-4-5 pricing).

### Layer 3 — Decision Governance (`helpers/decision_strategy.py`)

Two decisions arrive at this layer:
1. **Agent recommendation** from RiskAssessor (`normalize_decision_label`)
2. **Threshold decision** from ML score (`threshold_decision_from_score`)

Risk ordering: `APPROVE=0 < SOFT_DECLINE=1 < REVIEW=2 < BLOCK=3`

**Merge rule:** `final_decision = max(agent_rank, threshold_rank)`

The agent can only **escalate** the threshold decision, never de-escalate. This is conservative governance — an agent recommending APPROVE when the score says REVIEW results in REVIEW, not APPROVE.

**Failure fallback:** If the agent returns `None`, an empty string, or an unrecognised keyword, `threshold_decision_from_score` is used as the fallback and the error is logged.

---

## 5. Streamlit UI (`app.py`)

Loads on startup: `load_models()` (LightGBM + calibrator + feature names + AUC metric), `load_data()` (X_test, y_test, test_raw, hist_raw), `load_feature_importance()`.

### Tab 1 — Score Transaction

Flow per button click:
1. Fetch row at `test_idx` from `X_test` and `test_raw`
2. Run `transform_features_for_scoring` → aligned DataFrame
3. Layer 0: `rules_engine.evaluate(transaction_raw)` — checks hist_raw
4. Layer 1: `score_with_ml(transaction_features, models)` — always runs for score display
5. Gating: hard rule → skip to display; fast-path → skip to display; else → layers 2–3
6. Display: 5 metrics (score, risk, confidence, latency, cost) + correctness badge + SHAP bar chart + decision banner + agent pipeline steps

**SHAP bar chart:** top 8 features by absolute SHAP value, colored by direction (red = increases fraud risk, blue = decreases). Built with Plotly `px.bar` + diverging color scale centered at 0.

**Decision banner:** color-coded div (`_BANNER_STYLES` dict) rendered via `st.markdown(unsafe_allow_html=True)`.

**Agent pipeline steps:** 4 boxes showing each agent's name, a one-line note (score / finding summary / risk level / decision), and a badge (done / flagged). Flagged when `risk_level in (HIGH, CRITICAL)` or `decision_keyword in (BLOCK, REVIEW)`.

### Tab 2 — Batch Evaluation

Two code paths:

**ML only** (up to 2000 samples): `run_ml_batch()` → calibrated scores for random sample → `compute_batch_metrics()` → AUC, AP, precision, recall, F1, ROC curve arrays, PR curve arrays, threshold sweep DataFrame, confusion matrix values.

**With agents** (capped at 100 samples): `run_agent_batch()` runs the full pipeline per transaction with a progress callback. Returns rows with `fraud_score`, `agent_decision`, `threshold_decision`, `actual`, `confidence`, `latency_ms`, `risk_score`.

Charts rendered: score distribution histogram (legitimate vs fraud, overlaid), ROC curve, confusion matrix heatmap, PR curve, threshold sweep (precision/recall/F1/flag-rate vs threshold), agent decision bar chart, agent vs threshold crosstab heatmap, ML score vs agent risk score scatter, latency histogram, confidence box plot.

### Tab 3 — Analytics

In-session history only (cleared on page refresh). Accuracy is computed as:

```python
accuracy = ((decision in {BLOCK, REVIEW}) == (actual == 1)).mean()
```

This treats both BLOCK and REVIEW as "flagged" — correct if actual is fraud, incorrect if actual is legit.

### Tab 4 — Logs

Reads `logs/fraud_detection.log` line-by-line. Filter by level via string matching (`INFO in line` etc.). Download via `st.download_button`.

### Tab 5 — Settings

File existence checks for all 6 required artifacts. Feature importance bar chart (top 15 by gain) from `model/feature_importance.csv`.

---

## 6. Version 2 — Card-Based Split (`scripts/data_split_v2.py`)

### Why it's stricter than temporal

In v1 (temporal), a card that makes transactions in both January and March 2018 will have its January transactions in train and its March transactions in test. The model sees that card's transaction history, spending patterns, and fraud rate during training — so `card_fraud_rate`, `card_mean_amount` etc. all carry real information about the card at test time.

In v2 (card-based), every card appears in exactly one split. A card in the test set has **zero training history**. The model must generalise to completely unseen cardholders — a harder, more realistic evaluation of whether the model learned general fraud patterns vs memorized card-level history.

### How the split works

```
All unique card1 values (~N cards)
         │
         ├── Fraud cards (cards with ≥1 fraud txn) — stratify separately
         │       70% → train_cards
         │       10% → val_cards
         │       20% → test_cards
         │
         └── Clean cards — stratify separately
                 70% → train_cards
                 10% → val_cards
                 20% → test_cards

Transactions assigned by card:
  train  = rows where card1 ∈ train_cards  (+ rows where card1 is NaN)
  val    = rows where card1 ∈ val_cards
  test   = rows where card1 ∈ test_cards
```

Stratifying over fraud-card vs clean-card is necessary because fraud is card-concentrated. A naive shuffle could put a disproportionate fraction of fraud cards in one split, skewing fraud rates. After stratified split, all three sets have ~3.5% fraud rate.

**Guarantee enforced:** `assert len(train_cards ∩ val_cards) == 0` etc. — any leaking card raises an error immediately.

### Impact on feature engineering

| Feature | v1 (temporal) | v2 (card-based) |
|---|---|---|
| `card_total_txns` | >0 for many val/test cards (seen in train) | 0 for ALL val/test cards |
| `card_mean_amount` | Actual card mean for seen cards | Falls back to global train mean |
| `card_std_amount` | Actual card std | Falls back to global train std |
| `card_max_amount` | Actual card max | Falls back to global train 75th pct |
| `is_new_card` | 1 for ~5–10% of val/test rows | 1 for ~100% of val/test rows |
| `device_fraud_rate` | Unchanged — devices span cards | Unchanged |
| `email_fraud_rate` | Unchanged — domains span cards | Unchanged |

**`is_new_card`** (new feature in both v1 and v2): binary flag set to 1 when `card_total_txns == 0`. In v2, this is always 1 for val/test — it becomes a single clean signal that tells the model "you have no prior history on this cardholder." The model can then lean on device, email, address, and transaction features rather than card history.

**Global fallback (new in both v1 and v2):** unseen card features previously got 0, which was wrong — `card_mean_amount = 0` looks like a $0 spending history. Now they fall back to the global training-set mean/std, which is the correct prior when no card-specific history exists.

### What to expect in model metrics

| Metric | v1 expected | v2 expected | Why |
|---|---|---|---|
| AUC-ROC | ~0.898 | ~0.86–0.89 | Can't use card history for test cards |
| PR-AUC | ~0.536 | lower | Harder to detect fraud on unseen cards |
| `is_new_card` feature importance | low | high | Always fires on val/test — very informative |
| `card_mean_amount` importance | high | lower | Replaced by global mean — less signal |

v2 AUC being lower does not mean v2 is a worse model. It means v1's AUC was partially inflated by card-level memorisation. **v2 is a more honest estimate of real-world performance** where most transactions come from cards the model has never seen.

### Files produced

```
data/
├── ieee_prepared_v2.pkl          ← card-based splits (from data_split_v2.py)
├── features_train_v2.pkl         ← (X_train, y_train, feature_names)
├── features_val_v2.pkl
├── features_test_v2.pkl
└── train_statistics_v2.pkl       ← card/device/email/address stats from v2 train set
```

The model artifacts (`model/`) and the Streamlit app are version-agnostic — point `ml_pipeline.py` at the `_v2` pickle files and everything downstream works unchanged.
