from typing import Any

def normalize_decision_label(decision: Any) -> str:
    if decision is None:
        return "🟠 REVIEW"

    raw = str(decision).strip().upper()
    for marker in ["🔴", "🟠", "🟡", "🟢"]:
        raw = raw.replace(marker, "").strip()

    if raw in {"APPROVE", "APPROVED", "ALLOW"}:
        return "🟢 APPROVE"
    if raw in {"SOFT_DECLINE", "SOFT-DECLINE", "SOFT DECLINE", "SOFTDECLINE"}:
        return "🟡 SOFT_DECLINE"
    if raw == "REVIEW":
        return "🟠 REVIEW"
    if raw == "BLOCK":
        return "🔴 BLOCK"
    if "BLOCK" in raw:
        return "🔴 BLOCK"
    if "REVIEW" in raw:
        return "🟠 REVIEW"
    if "SOFT" in raw and "DECLINE" in raw:
        return "🟡 SOFT_DECLINE"
    if "APPROVE" in raw or "ALLOW" in raw:
        return "🟢 APPROVE"
    return "🟠 REVIEW"


DECISION_RISK_ORDER = {
    "APPROVE": 0,
    "SOFT_DECLINE": 1,
    "REVIEW": 2,
    "BLOCK": 3,
}


def threshold_decision_from_score(
    fraud_score: float,
    block_threshold: float,
    review_threshold: float,
    soft_decline_threshold: float,
) -> str:
    if fraud_score >= block_threshold:
        return "🔴 BLOCK"
    if fraud_score >= review_threshold:
        return "🟠 REVIEW"
    if fraud_score >= soft_decline_threshold:
        return "🟡 SOFT_DECLINE"
    return "🟢 APPROVE"


def extract_decision_keyword(decision_label: str) -> str:
    return normalize_decision_label(decision_label).split()[-1]


def decision_risk_level(decision_label: str) -> int:
    keyword = extract_decision_keyword(decision_label)
    return DECISION_RISK_ORDER.get(keyword, 2)


def merge_agent_and_threshold_decision(agent_label: str, threshold_label: str) -> str:
    agent_risk = decision_risk_level(agent_label)
    threshold_risk = decision_risk_level(threshold_label)
    return threshold_label if threshold_risk > agent_risk else agent_label


def fallback_explanation_decision(decision_keyword: str, fraud_score: float) -> str:
    if decision_keyword == "BLOCK":
        return f"This transaction has been BLOCKED due to very high fraud risk ({fraud_score:.1%}). The model detected critical fraud indicators."
    if decision_keyword == "REVIEW":
        return f"This transaction is flagged for REVIEW due to elevated fraud risk ({fraud_score:.1%}). Manual investigation is recommended."
    if decision_keyword == "SOFT_DECLINE":
        return f"This transaction requires verification ({fraud_score:.1%} fraud risk). Please complete 3D Secure verification to proceed."
    return f"This transaction is APPROVED. Low fraud risk detected ({fraud_score:.1%})."
