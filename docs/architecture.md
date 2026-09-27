 # Fraud Investigation Engine — Architecture Diagrams

> **Miro import tip:** In Miro, add a new frame → Insert → Mermaid diagram → paste any block below.

---

## 1. System Overview

```mermaid
graph TB
    subgraph UI["🖥️  Streamlit UI (app.py)"]
        T1[Score Transaction]
        T2[Batch Evaluation]
        T3[Analytics]
        T4[Logs]
        T5[Settings]
    end

    subgraph Data["📦  Data Layer"]
        D1[(IEEE-CIS Dataset\n590k transactions)]
        D2[(LightGBM Model\n843 trees, 460 features)]
        D3[(Calibrator\nIsotonic Regression)]
        D4[(Thresholds\nthresholds.json)]
        D5[(Feature Names\n460 columns)]
    end

    subgraph Pipeline["⚙️  Decision Pipeline"]
        L0[Layer 0\nHard Rules Engine]
        L1[Layer 1\nML Scoring]
        L2[Layer 2\nAgentic Investigation]
        L3[Layer 3\nDecision Merge]
    end

    subgraph Agents["🤖  Agent Orchestrator (investigator.py)"]
        A1[MLAnalyst]
        A2[TransactionInvestigator]
        A3[RiskAssessor]
        A4[DecisionExplainer]
    end

    subgraph Tools["🔍  Investigation Tools (queries.py)"]
        Q1[CardVelocityChecker]
        Q2[DeviceProfileChecker]
        Q3[EmailDomainChecker]
        Q4[AddressClusterChecker]
    end

    subgraph External["☁️  External Services"]
        E1[Anthropic Claude API\nHaiku-4-5]
    end

    T1 --> Pipeline
    T2 --> Pipeline
    Data --> Pipeline

    Pipeline --> L0
    L0 -->|"Rule triggered → BLOCK"| L3
    L0 -->|"No rule"| L1
    L1 -->|"score ≥ 0.72 → fast-path BLOCK"| L3
    L1 -->|"score < 0.01 → fast-path APPROVE"| L3
    L1 -->|"0.01 – 0.72\nuncertain zone"| L2
    L2 --> Tools
    L2 --> Agents
    Agents --> E1
    Agents --> L3

    D2 --> L1
    D3 --> L1
    D4 --> L1
    D5 --> L1
    D1 --> Tools
```

---

## 2. Layered Decision Flow

```mermaid
flowchart TD
    IN([Transaction Input]) --> R0

    subgraph LAYER0["Layer 0 — Hard Rules  ⚡ deterministic"]
        R0{Blacklisted\nDevice?}
        R1{Card Velocity\n≥3/min or ≥10/hr?}
        R2{Large Amount\n>$500 on new device?}
        R0 -->|Yes| BLOCK0([🔴 BLOCK])
        R0 -->|No| R1
        R1 -->|Yes| BLOCK1([🔴 BLOCK])
        R1 -->|No| R2
        R2 -->|Yes| BLOCK2([🔴 BLOCK])
        R2 -->|No| ML
    end

    subgraph LAYER1["Layer 1 — ML Fast-Path  ⚡ <100ms"]
        ML[LightGBM\n460 features\n843 trees]
        ML --> CAL[Isotonic Calibrator\n→ calibrated score 0–1]
        CAL --> FP{Fast-path\ncheck}
        FP -->|score ≥ 0.72| BLOCK3([🔴 BLOCK\nfast-path])
        FP -->|score < 0.01| APPROVE([🟢 APPROVE\nfast-path])
        FP -->|0.01 – 0.72| SEL
    end

    subgraph LAYER2["Layer 2 — Agentic Investigation  🤖 ~500ms–2s"]
        SEL[AdaptiveToolSelector\n— score-gated —]
        SEL -->|"score 0.2–0.8\n(2000ms budget)"| TOOLS
        SEL -->|"score > 0.8 or < 0.2\n(skip tools)"| AGENTS

        subgraph TOOLS["Investigation Tools"]
            T1[CardVelocityChecker]
            T2[DeviceProfileChecker]
            T3[EmailDomainChecker]
            T4[AddressClusterChecker]
        end

        TOOLS --> AGENTS

        subgraph AGENTS["4-Agent Chain  Claude Haiku"]
            A1[MLAnalyst\n→ interpret score + SHAP]
            A2[TransactionInvestigator\n→ tool signal analysis]
            A3[RiskAssessor\n→ synthesize risk]
            A4[DecisionExplainer\n→ customer explanation]
            A1 --> A2 --> A3 --> A4
        end

        AGENTS --> AREC[Agent Recommendation\nBLOCK / REVIEW / SOFT_DECLINE / APPROVE]
    end

    subgraph LAYER3["Layer 3 — Decision Governance"]
        AREC --> MERGE[merge_agent_and_threshold_decision\n— takes max risk —]
        CAL --> TH[threshold_decision_from_score\nblock=0.53 / review=0.21 / soft_decline=0.07]
        TH --> MERGE
        MERGE --> FINAL([Final Decision])
    end

    FINAL --> OUT([Output\nDecision + Explanation\n+ SHAP + Cost + Latency])

    style LAYER0 fill:#fff3e0,stroke:#ff9800
    style LAYER1 fill:#e3f2fd,stroke:#2196f3
    style LAYER2 fill:#f3e5f5,stroke:#9c27b0
    style LAYER3 fill:#e8f5e9,stroke:#4caf50
```

---

## 3. Agent Pipeline Detail

```mermaid
sequenceDiagram
    participant O as FraudInvestigationOrchestrator
    participant TS as AdaptiveToolSelector
    participant TE as ToolExecutor
    participant CV as CardVelocityChecker
    participant DP as DeviceProfileChecker
    participant EM as EmailDomainChecker
    participant AC as AddressClusterChecker
    participant ML as MLAnalyst
    participant TI as TransactionInvestigator
    participant RA as RiskAssessor
    participant DE as DecisionExplainer
    participant C as Claude API (Haiku)

    O->>TS: select_tools(fraud_score=0.35)
    TS-->>O: [velocity, device, email, address]

    O->>TE: execute_selected(tools, transaction)
    par Tool execution (parallel budget 2000ms)
        TE->>CV: check(card_id, hist_raw)
        CV-->>TE: ToolResult(velocities, signals)
        TE->>DP: check(device_id, hist_raw)
        DP-->>TE: ToolResult(device_stats, signals)
        TE->>EM: check(email_domain, hist_raw)
        EM-->>TE: ToolResult(domain_type, signals)
        TE->>AC: check(billing_addr, hist_raw)
        AC-->>TE: ToolResult(cluster_stats, signals)
    end
    TE-->>O: tool_results dict

    O->>ML: analyze(score, shap, features)
    ML->>C: system prompt + score context
    C-->>ML: {fraud_score, confidence, top_features, summary}
    ML-->>O: ml_analysis

    O->>TI: investigate(transaction, tool_results, ml_analysis)
    TI->>C: system prompt + signals
    C-->>TI: {fraud_signals, overall_risk, summary}
    TI-->>O: investigation

    O->>RA: assess(transaction, ml_analysis, investigation)
    RA->>C: system prompt + all evidence
    C-->>RA: {risk_level, risk_score, confidence, recommendation}
    RA-->>O: risk_assessment

    O->>DE: explain(transaction, risk_assessment)
    DE->>C: system prompt + decision
    C-->>DE: customer_explanation (plain text)
    DE-->>O: explanation

    O-->>O: build FraudAssessment(decision, cost_usd, latency_ms)
```

---

## 4. Data Flow & Leakage Boundaries

```mermaid
graph LR
    subgraph RAW["Raw IEEE-CIS Data"]
        TR_RAW[train_transaction.csv\ntrain_identity.csv]
        TE_RAW[test_transaction.csv\ntest_identity.csv]
    end

    subgraph PREP["data_prep.ipynb\nTemporal Split"]
        TR[Train\n413k rows]
        VA[Val\n84k rows]
        TE[Test\n87k rows]
        TR_RAW --> TR & VA & TE
    end

    subgraph FE["feature_engineering\n460 features"]
        direction TB
        FE_TRAIN["Fit on TRAIN ONLY\n— group stats\n— thresholds\n— outlier caps\n— category maps"]
        FE_APPLY["Apply to ALL splits\n(no train-time data)"]
        TR --> FE_TRAIN
        FE_TRAIN --> FE_APPLY
        VA & TE --> FE_APPLY
    end

    subgraph ML_TRAIN["ml_pipeline.py  (offline)"]
        LGB[LightGBM\ntrained on TRAIN]
        CAL[Isotonic Calibrator\nfitted on VAL]
        THR[Threshold Optimizer\nderived on VAL]
        FE_APPLY --> LGB
        LGB --> CAL
        CAL --> THR
    end

    subgraph ARTIFACTS["model/ artifacts"]
        M1[lightgbm_model.pkl]
        M2[calibrator.pkl]
        M3[feature_names.pkl]
        M4[thresholds.json]
        LGB --> M1
        CAL --> M2
        FE_TRAIN --> M3
        THR --> M4
    end

    subgraph HIST["Historical Context for Tools"]
        HIST_DATA["hist_raw = TRAIN + VAL only\n497k rows\n(no TEST leakage)"]
        TR --> HIST_DATA
        VA --> HIST_DATA
    end

    subgraph INFERENCE["Runtime Inference"]
        TX[New Transaction]
        FEAT[transform_features_for_scoring]
        SCORE[score_with_ml]
        TOOLS[Investigation Tools\n— query HIST only —]
        TX --> FEAT
        M3 --> FEAT
        FEAT --> SCORE
        M1 & M2 --> SCORE
        M4 --> SCORE
        TX --> TOOLS
        HIST_DATA --> TOOLS
    end

    style HIST fill:#fff9c4,stroke:#f9a825
    style INFERENCE fill:#e8f5e9,stroke:#388e3c
```

---

## 5. Component Dependency Map

```mermaid
graph TD
    APP[app.py\nStreamlit UI]

    subgraph HELPERS
        LDR[helpers/loaders.py]
        BST[helpers/decision_strategy.py]
        RUL[helpers/rules_engine.py]
        BAT[helpers/batch_runner.py]
        SHP[helpers/feature_attribution.py]
    end

    subgraph AGENTS_MOD
        INV[agents/investigator.py\nOrchestrator + 4 Agents]
        TEX[agents/tool_execution.py]
        TSL[agents/tool_selection.py]
        DLG[agents/dialogue.py]
        LRN[agents/learning.py]
    end

    subgraph TOOLS_MOD
        QRY[tools/queries.py\n4 Investigation Tools]
    end

    subgraph SCRIPTS
        MLP[scripts/ml_pipeline.py\nscore_with_ml()]
        FEN[scripts/feature_engineering.py]
        FIA[scripts/fraud_investigation_agents.py]
    end

    subgraph CONFIG
        CFG[config.py]
        THR[model/thresholds.json]
    end

    APP --> LDR
    APP --> BST
    APP --> RUL
    APP --> BAT
    APP --> SHP
    APP --> FIA
    APP --> MLP
    APP --> CFG

    LDR --> MLP
    BAT --> MLP
    BAT --> INV
    FIA --> TSL
    FIA --> TEX
    FIA --> INV

    INV --> DLG
    INV --> LRN
    TEX --> QRY
    TSL --> QRY

    CFG --> THR
    MLP --> CFG
    INV --> CFG

    style APP fill:#bbdefb,stroke:#1976d2
    style HELPERS fill:#f3e5f5,stroke:#7b1fa2
    style AGENTS_MOD fill:#e8f5e9,stroke:#388e3c
    style TOOLS_MOD fill:#fff3e0,stroke:#f57c00
    style SCRIPTS fill:#fce4ec,stroke:#c62828
    style CONFIG fill:#eceff1,stroke:#546e7a
```

---

## 6. Threshold Decision Zones

```mermaid
xychart-beta
    title "Decision Zones by Fraud Score"
    x-axis ["0.00", "0.01", "0.07", "0.21", "0.53", "0.72", "1.00"]
    y-axis "Zone" 0 --> 4
    bar [0, 1, 2, 3, 4, 4, 4]
```

| Score Range | Zone | Decision | Agent Invoked? |
|---|---|---|---|
| `< 0.01` | Fast-path Approve | 🟢 APPROVE | No |
| `0.01 – 0.07` | Low risk | 🟢 APPROVE | Yes (if in 0.01–0.72) |
| `0.07 – 0.21` | Elevated risk | 🟡 SOFT_DECLINE | Yes |
| `0.21 – 0.53` | Medium risk | 🟠 REVIEW | Yes |
| `0.53 – 0.72` | High risk | 🔴 BLOCK | Yes |
| `≥ 0.72` | Fast-path Block | 🔴 BLOCK | No |
