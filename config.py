import json
from pathlib import Path

# ── Decision thresholds ──────────────────────────────────────────────────────
# Loaded from model/thresholds.json (produced by notebooks/model_training.ipynb
# Section 13 — Threshold Optimisation).  Hardcoded values are fallbacks only.
#
# Derivation criteria (precision-anchored on the validation set):
#   fastpath_approve  val precision < 10%  → auto-approve, skip agents
#   soft_decline      val precision ≥ 30%  → soft-challenge user
#   review            val precision ≥ 60%  → route to human analyst
#   block             val precision ≥ 80%  → auto-block
#   fastpath_block    val precision ≥ 90%  → auto-block, skip agents

_THRESHOLD_DEFAULTS = {
    "fastpath_approve": 0.01,
    "soft_decline":     0.07,
    "review":           0.21,
    "block":            0.53,
    "fastpath_block":   0.72,
}

_threshold_file = Path("model/thresholds.json")
if _threshold_file.exists():
    with open(_threshold_file) as _f:
        _t = json.load(_f)
    BLOCK_THRESHOLD        = _t.get("block",            _THRESHOLD_DEFAULTS["block"])
    REVIEW_THRESHOLD       = _t.get("review",           _THRESHOLD_DEFAULTS["review"])
    SOFT_DECLINE_THRESHOLD = _t.get("soft_decline",     _THRESHOLD_DEFAULTS["soft_decline"])
    ML_FASTPATH_BLOCK      = _t.get("fastpath_block",   _THRESHOLD_DEFAULTS["fastpath_block"])
    ML_FASTPATH_APPROVE    = _t.get("fastpath_approve", _THRESHOLD_DEFAULTS["fastpath_approve"])
else:
    BLOCK_THRESHOLD        = _THRESHOLD_DEFAULTS["block"]
    REVIEW_THRESHOLD       = _THRESHOLD_DEFAULTS["review"]
    SOFT_DECLINE_THRESHOLD = _THRESHOLD_DEFAULTS["soft_decline"]
    ML_FASTPATH_BLOCK      = _THRESHOLD_DEFAULTS["fastpath_block"]
    ML_FASTPATH_APPROVE    = _THRESHOLD_DEFAULTS["fastpath_approve"]

# ── Chart color map ──────────────────────────────────────────────────────────
DECISION_COLOR: dict[str, str] = {
    "APPROVE":       "#4CAF50",
    "SOFT_DECLINE":  "#FF9800",
    "REVIEW":        "#2196F3",
    "BLOCK":         "#F44336",
    "UNKNOWN":       "#9E9E9E",
    "ERROR":         "#795548",
}


MOCK_MODE = False  # Set to True to use mock data and skip API calls (for local testing)
VALID_DECISIONS = {"APPROVE", "SOFT_DECLINE", "REVIEW", "BLOCK"}
VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}



# ── Claude API pricing (claude-haiku-4-5) ────────────────────────────────────
CLAUDE_INPUT_COST_PER_TOKEN  = 0.80 / 1_000_000   # $0.80 per 1M input tokens
CLAUDE_OUTPUT_COST_PER_TOKEN = 4.00 / 1_000_000   # $4.00 per 1M output tokens

# ── Logging paths ────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "fraud_detection.log"


def score_to_label(score: float) -> str:
    """Map a calibrated fraud score to a decision label."""
    if score >= BLOCK_THRESHOLD:
        return "BLOCK"
    if score >= REVIEW_THRESHOLD:
        return "REVIEW"
    if score >= SOFT_DECLINE_THRESHOLD:
        return "SOFT_DECLINE"
    return "APPROVE"
