from pathlib import Path

# ── Decision thresholds ──────────────────────────────────────────────────────
BLOCK_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.65
SOFT_DECLINE_THRESHOLD = 0.50

# ── Chart color map ──────────────────────────────────────────────────────────
DECISION_COLOR: dict[str, str] = {
    "APPROVE":       "#4CAF50",
    "SOFT_DECLINE":  "#FF9800",
    "REVIEW":        "#2196F3",
    "BLOCK":         "#F44336",
    "UNKNOWN":       "#9E9E9E",
    "ERROR":         "#795548",
}

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
