"""
FRAUD INVESTIGATION AGENTS
4 specialized Claude agents working together to investigate transactions

Uses:
- Claude API for agent reasoning (chain-of-thought)
- Custom tools for fraud investigation
- Multi-agent collaboration for comprehensive assessment
"""

import json
import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import config
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langsmith import traceable
    _LANGSMITH_ENABLED = bool(os.getenv("LANGSMITH_API_KEY"))
except ImportError:
    _LANGSMITH_ENABLED = False
    def traceable(_fn=None, **kwargs):
        if _fn is not None:
            return _fn
        return lambda fn: fn

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

# Add option for local testing with mock data (no api calls)
# take these from config.py

MOCK_MODE = config.MOCK_MODE

VALID_DECISIONS = config.VALID_DECISIONS
VALID_RISK_LEVELS = config.VALID_RISK_LEVELS


def _extract_json(text: str) -> str:
    """Return the JSON object substring from a model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        if result != result or result is None:  # NaN check
            return default
        return max(0.0, min(1.0, result))
    except Exception:
        return default


def normalize_recommendation(value: Any) -> str:
    if value is None:
        return "REVIEW"

    text = str(value).strip().upper()
    for marker in ["🔴", "🟠", "🟡", "🟢"]:
        text = text.replace(marker, "").strip()

    text = text.replace("-", "_").replace(" ", "_")
    if text in {"APPROVE", "APPROVED", "ALLOW"}:
        return "APPROVE"
    if text in {"SOFT_DECLINE", "SOFTDECLINE", "SOFT_DECLINE"}:
        return "SOFT_DECLINE"
    if text == "REVIEW":
        return "REVIEW"
    if text == "BLOCK":
        return "BLOCK"
    if "BLOCK" in text:
        return "BLOCK"
    if "REVIEW" in text:
        return "REVIEW"
    if "SOFT" in text and "DECLINE" in text:
        return "SOFT_DECLINE"
    if "APPROVE" in text or "ALLOW" in text:
        return "APPROVE"
    return "REVIEW"


def validate_ml_analysis(ml_analysis: Any) -> Dict[str, Any]:
    if not isinstance(ml_analysis, dict):
        logger.warning("ML analysis is not a dict; normalizing to defaults")
        ml_analysis = {}

    normalized = {
        "fraud_score": _safe_float(ml_analysis.get("fraud_score"), 0.5),
        "calibrated_probability": _safe_float(ml_analysis.get("calibrated_probability"), _safe_float(ml_analysis.get("fraud_score"), 0.5)),
        "confidence": _safe_float(ml_analysis.get("confidence"), 0.5),
        "top_features": ml_analysis.get("top_features", []),
        "summary": str(ml_analysis.get("summary", "")),
        "caveats": str(ml_analysis.get("caveats", "")),
        "error": ml_analysis.get("error"),
    }

    if not isinstance(normalized["top_features"], list):
        logger.warning("ML analysis top_features is not a list; replacing with empty list")
        normalized["top_features"] = []

    normalized.update({k: v for k, v in ml_analysis.items() if k not in normalized})
    return normalized


def validate_risk_assessment(risk_assessment: Any) -> Dict[str, Any]:
    if not isinstance(risk_assessment, dict):
        risk_assessment = {}

    recommendation = normalize_recommendation(risk_assessment.get("recommendation"))
    risk_level = str(risk_assessment.get("risk_level", "MEDIUM")).strip().upper()
    for marker in ["🔴", "🟠", "🟡", "🟢"]:
        risk_level = risk_level.replace(marker, "").strip()
    risk_level = risk_level.replace("-", "_").replace(" ", "_")
    if risk_level not in VALID_RISK_LEVELS:
        if "CRITICAL" in risk_level:
            risk_level = "CRITICAL"
        elif "HIGH" in risk_level:
            risk_level = "HIGH"
        elif "LOW" in risk_level:
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"

    normalized = {
        "recommendation": recommendation,
        "risk_level": risk_level,
        "risk_score": _safe_float(risk_assessment.get("risk_score"), 0.5),
        "confidence": _safe_float(risk_assessment.get("confidence"), 0.5),
        "key_signals": risk_assessment.get("key_signals", []),
        "contradicting_signals": risk_assessment.get("contradicting_signals", []),
        "ml_contribution": _safe_float(risk_assessment.get("ml_contribution"), 0.0),
        "investigator_contribution": _safe_float(risk_assessment.get("investigator_contribution"), 0.0),
        "recommendation_reason": str(risk_assessment.get("reasoning", "")),
        "error": risk_assessment.get("error"),
    }

    normalized.update({k: v for k, v in risk_assessment.items() if k not in normalized})
    return normalized

@dataclass
class AgentMessage:
    """Message between agents"""
    agent_name: str
    role: str  # 'user', 'assistant'
    content: str
    thinking: Optional[str] = None  # For internal reasoning

@dataclass
class FraudAssessment:
    """Final fraud assessment output"""
    transaction_id: str
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    risk_score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    
    ml_analysis: Dict[str, Any]
    investigation_findings: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    customer_explanation: str
    
    decision: str  # 'APPROVE', 'SOFT_DECLINE', 'REVIEW', 'BLOCK'
    reasoning: str
    reasoning_details: Dict[str, Any]
    execution_time_ms: float
    agent_cost_usd: float = 0.0

# ============================================================================
# AGENT 1: ML ANALYST
# ============================================================================

class MLAnalyst:
    """
    Interprets ML model outputs and explains predictions using SHAP values
    """

    SYSTEM_PROMPT = """You are an ML Analyst specializing in fraud risk scoring.

Your role:
- Interpret ML model outputs (fraud score, calibrated probability, SHAP values)
- Explain which features drove the prediction
- Quantify the confidence in the score
- Flag when model confidence is low

You have access to:
- fraud_score: Raw model output (0-1, higher = more likely fraud)
- probability: Calibrated probability (adjusted for bias)
- shap_values: Feature importance breakdown
- feature_names: Names of features used

Output format (JSON):
{
    "fraud_score": 0.72,
    "calibrated_probability": 0.70,
    "confidence": 0.85,
    "top_features": [
        {"feature": "velocity_1h", "impact": 0.18, "direction": "increases_fraud"},
        {"feature": "device_age_days", "impact": -0.12, "direction": "decreases_fraud"}
    ],
    "summary": "The model predicts 0.72 fraud probability with 0.85 confidence...",
    "caveats": "Model confidence is high because features align, no contradictions"
}

Always:
1. State the fraud score and confidence level
2. List top 3-5 driving features with their impact
3. Explain what each feature means in plain English
4. Note if any features contradict each other
5. Flag if model uncertainty is high
"""
    
    def __init__(self, client: Any):
        self.client = client
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

    @traceable(name="MLAnalyst", run_type="llm")
    def analyze(self, fraud_score: float, shap_values: Dict[str, float],
                feature_importance: List[tuple],
                raw_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze ML model output and provide interpretation
        """
        
        # Format SHAP values for Claude
        top_features = feature_importance[:5]
        feature_lines = []
        for fname, impact in top_features:
            try:
                impact_value = float(impact)
            except Exception:
                impact_value = float(str(impact).strip())
            feature_lines.append(f"  - {fname}: {impact_value:.4f}")
        features_text = "\n".join(feature_lines)
        
        raw_line = (
            f"Raw Model Output (LightGBM): {raw_score:.4f}\n"
            if raw_score is not None else ""
        )
        user_message = f"""Analyze this fraud detection model output:

{raw_line}Calibrated Probability (isotonic calibration): {fraud_score:.4f}

Top features driving prediction:
{features_text}

Provide your analysis as JSON."""
        
        if MOCK_MODE:
            # Return mock analysis for testing
            mock_analysis = {
                "fraud_score": fraud_score,
                "calibrated_probability": fraud_score,
                "confidence": 0.85,
                "top_features": [
                    {"feature": "velocity_1h", "impact": 0.18, "direction": "increases_fraud"},
                    {"feature": "device_age_days", "impact": -0.12, "direction": "decreases_fraud"}
                ],
                "summary": f"The model predicts {fraud_score:.2f} fraud probability with 0.85 confidence based on top features.",
                "caveats": "Model confidence is high because features align, no contradictions",
                "raw_response": "MOCK RESPONSE"
            }
            return validate_ml_analysis(mock_analysis)
        
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            temperature=0,
            max_tokens=1000,
            system=self.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "{"},
            ]
        )
        self.last_usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        response_text = "{" + response.content[0].text

        # Parse response
        try:
            raw_analysis = json.loads(_extract_json(response_text))
            raw_analysis["raw_response"] = response_text
            return validate_ml_analysis(raw_analysis)
        except Exception as exc:
            logger.warning("MLAnalyst parse failed; using fallback analysis: %s", exc)
            fallback = {
                'fraud_score': fraud_score,
                'calibrated_probability': fraud_score,
                'confidence': 0.5,
                'summary': response_text,
                'error': 'Could not parse JSON response',
                'raw_response': response_text
            }
            return validate_ml_analysis(fallback)

# ============================================================================
# AGENT 2: TRANSACTION INVESTIGATOR
# ============================================================================

class TransactionInvestigator:
    """
    Investigates transaction details using tools and detects fraud patterns
    """
    
    SYSTEM_PROMPT = """You are a Transaction Investigator specializing in fraud detection.

Your role:
- Investigate this specific transaction in detail
- Analyze provided tool results for fraud patterns
- Detect card testing, account takeover, location anomalies
- Synthesize findings into coherent fraud assessment

Fraud patterns to detect:

1. CARD TESTING
   - Multiple small transactions ($5-$50) in rapid succession
   - Amounts incrementally increasing (test different limits)
   - Different merchants/countries within minutes
   - Goal: Test card validity

2. ACCOUNT TAKEOVER
   - Rapid spending spike compared to account history
   - New device or location
   - Multiple high-value transactions
   - Different merchants than usual

3. LOCATION ANOMALIES
   - Impossible travel (two txns far apart, minutes apart)
   - First use in new country
   - Unusual time of day for location

4. DEVICE ANOMALIES
   - New device
   - Device with high fraud rate
   - Multiple cards on same device

Output format (JSON):
{
    "card_velocity": {...tool result...},
    "device_profile": {...tool result...},
    "email_domain": {...tool result...},
    "address_cluster": {...tool result...},
    
    "fraud_signals": [
        {"signal": "card_testing", "severity": "high", "evidence": "..."},
        {"signal": "new_device", "severity": "medium", "evidence": "..."}
    ],
    "contradicting_signals": [],
    "summary": "Investigation summary...",
    "overall_risk": "high"
}

Always:
1. Present findings from each tool
2. Extract fraud signals with severity
3. Note contradicting signals
4. Explain pattern recognition
"""
    
    def __init__(self, client: Any):
        self.client = client
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

    @traceable(name="TransactionInvestigator", run_type="llm")
    def investigate(self, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Investigate transaction using tool results
        """
        
        # Format tool results for Claude
        tools_summary = json.dumps(tool_results, indent=2, default=str)
        
        user_message = f"""Investigate this transaction based on tool results:

{tools_summary}

Provide your investigation as JSON with:
- fraud_signals: list of detected patterns
- contradicting_signals: any opposing evidence
- summary: narrative explanation
- overall_risk: LOW, MEDIUM, HIGH, CRITICAL"""
        
        if MOCK_MODE:
            # Return mock investigation for testing
            return {
                "fraud_signals": [
                    {"signal": "card_testing", "severity": "high", "evidence": "Multiple small transactions in rapid succession"}
                ],
                "contradicting_signals": [],
                "summary": "Investigation summary...",
                "overall_risk": "HIGH",
                "raw_response": "MOCK RESPONSE"
            }

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            temperature=0,
            max_tokens=1500,
            system=self.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "{"},
            ]
        )
        self.last_usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        response_text = "{" + response.content[0].text

        try:
            result = json.loads(_extract_json(response_text))
            result["raw_response"] = response_text
            return result
        except Exception:
            return {
                'fraud_signals': [],
                'summary': response_text,
                'error': 'Could not parse JSON response',
                'overall_risk': 'MEDIUM',
                'raw_response': response_text
            }

# ============================================================================
# AGENT 3: RISK ASSESSOR
# ============================================================================

class RiskAssessor:
    """
    Synthesizes all evidence into final fraud risk assessment
    """
    
    SYSTEM_PROMPT = """You are a Risk Assessment expert synthesizing fraud signals.

Your role:
- Combine ML scores with investigator findings
- Make final risk judgment (LOW, MEDIUM, HIGH, CRITICAL)
- Quantify confidence in assessment
- Identify key risk factors

Risk assessment framework:

CRITICAL (>0.85):
- Multiple strong fraud signals (3+) aligned
- Card velocity + new device + impossible travel
- ML score >0.80 with high confidence
→ Action: BLOCK immediately

HIGH (0.65-0.85):
- 2+ fraud signals present
- ML score >0.75 with medium confidence
- Card testing + new device
→ Action: SOFT_DECLINE or REVIEW

MEDIUM (0.40-0.65):
- 1 strong signal OR 2+ weak signals
- ML score 0.50-0.75
- Slight velocity increase, normal device
→ Action: REVIEW or 3DS challenge

LOW (<0.40):
- No fraud signals
- ML score <0.50 with high confidence
- Normal behavior
→ Action: APPROVE

Output format (JSON):
{
    "risk_level": "HIGH",
    "risk_score": 0.78,
    "confidence": 0.88,
    "key_signals": ["velocity", "device"],
    "contradicting_signals": [],
    "ml_contribution": 0.72,
    "investigator_contribution": 0.80,
    "recommendation": "SOFT_DECLINE",
    "reasoning": "..."
}

Always:
1. Use conjunction (all must align for high confidence)
2. Weight ML score heavily
3. Respect investigator findings
4. Explain reasoning clearly
5. Quantify confidence
"""
    
    def __init__(self, client: Any):
        self.client = client
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

    @traceable(name="RiskAssessor", run_type="llm")
    def assess(self, ml_analysis: Dict[str, Any],
               investigation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize ML and investigation findings into risk assessment
        """
        
        context = f"""ML Analysis:
{json.dumps(ml_analysis, indent=2, default=str)}

Investigation Findings:
{json.dumps(investigation, indent=2, default=str)}

Provide risk assessment as JSON with:
- risk_level: LOW, MEDIUM, HIGH, CRITICAL
- risk_score: 0.0-1.0
- confidence: 0.0-1.0
- recommendation: APPROVE, SOFT_DECLINE, REVIEW, BLOCK"""
        
        if MOCK_MODE:
            # Return mock assessment for testing
            mock_assessment = {
                "risk_level": "HIGH",
                "risk_score": 0.78,
                "confidence": 0.88,
                "key_signals": ["velocity", "device"],
                "contradicting_signals": [],
                "ml_contribution": ml_analysis.get('fraud_score', 0.72),
                "investigator_contribution": 0.80,
                "recommendation": "SOFT_DECLINE",
                "reasoning": "ML score is high and investigation found strong velocity signal.",
                "raw_response": "MOCK RESPONSE"
            }
            return validate_risk_assessment(mock_assessment)

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            temperature=0,
            max_tokens=1200,
            system=self.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": context},
                {"role": "assistant", "content": "{"},
            ]
        )
        self.last_usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        response_text = "{" + response.content[0].text

        try:
            raw_assessment = json.loads(_extract_json(response_text))
            raw_assessment["raw_response"] = response_text
            return validate_risk_assessment(raw_assessment)
        except Exception as exc:
            logger.warning("RiskAssessor parse failed; using fallback assessment: %s", exc)
            fallback = {
                'risk_level': 'MEDIUM',
                'risk_score': 0.5,
                'confidence': 0.5,
                'recommendation': 'REVIEW',
                'error': 'Could not parse JSON response',
                'raw_response': response_text
            }
            return validate_risk_assessment(fallback)

# ============================================================================
# AGENT 4: DECISION EXPLAINER
# ============================================================================

class DecisionExplainer:
    """
    Translates technical decision to customer-friendly explanation
    """
    
    SYSTEM_PROMPT = """You are a Decision Explainer translating fraud decisions to plain English.

Your role:
- Take technical fraud assessment
- Translate to customer-friendly language
- Explain decision clearly and respectfully
- Suggest next actions

Explanation guidelines:

For BLOCK:
"We've declined your $X transaction. We detected signs of unauthorized use:
- [specific signal]
- [specific signal]
For your security, we've blocked the transaction.
Action: Call us at [number] to verify this was you."

For SOFT_DECLINE:
"We need to verify this is you. We noticed:
- [specific observation]
Please complete 3D Secure verification to approve."

For REVIEW:
"Your transaction is being reviewed by our fraud team.
We may contact you within 24 hours if we need more information.
Reason: [specific reason]"

For APPROVE:
"Transaction approved. We detected no suspicious activity."

Key principles:
1. Customer-friendly language (no jargon)
2. Explain specific concerns
3. Provide clear next steps
4. Reassure about security

Output as plain English explanation (not JSON).
"""
    
    def __init__(self, client: Any):
        self.client = client
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

    @traceable(name="DecisionExplainer", run_type="llm")
    def explain(self, decision: str, risk_assessment: Dict[str, Any],
                transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate customer-friendly explanation
        """
        
        context = f"""Generate a customer-friendly explanation for this fraud decision:

Decision: {decision}
Risk Level: {risk_assessment.get('risk_level')}
Key Signals: {risk_assessment.get('key_signals')}
Transaction Amount: ${transaction.get('Amount', 'unknown')}

Write a clear, respectful explanation in plain English (no jargon)."""
        
        if MOCK_MODE:
            return {
                "explanation": f"We need to verify this is you. We noticed some unusual activity on your account, including {', '.join(risk_assessment.get('key_signals', []))}. Please complete 3D Secure verification to approve this transaction.",
                "raw_response": "MOCK RESPONSE"
            }
        
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            temperature=0,
            max_tokens=500,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}]
        )
        self.last_usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}

        return {
            "explanation": response.content[0].text,
            "raw_response": response.content[0].text
        }

# ============================================================================
# ORCHESTRATOR (Runs all 4 agents)
# ============================================================================



class FraudInvestigationOrchestrator:
    """
    Orchestrates all 4 agents working together
    """
    
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
        if api_key:
            if anthropic is None:
                raise RuntimeError("The anthropic package is not installed. Install it to use real LLM calls.")
            self.client = anthropic.Anthropic(api_key=api_key)
        elif auth_token:
            if anthropic is None:
                raise RuntimeError("The anthropic package is not installed. Install it to use real LLM calls.")
            self.client = anthropic.Anthropic(auth_token=auth_token)
        else:
            if not config.MOCK_MODE:
                raise RuntimeError(
                    "No Anthropic credentials found. Set ANTHROPIC_API_KEY in your .env file, "
                    "or set MOCK_MODE = True in config.py to run without real API calls."
                )
            global MOCK_MODE
            MOCK_MODE = True
            self.client = None

        self.ml_analyst = MLAnalyst(self.client)
        self.investigator = TransactionInvestigator(self.client)
        self.risk_assessor = RiskAssessor(self.client)
        self.explainer = DecisionExplainer(self.client)
    
    @traceable(name="FraudInvestigation", run_type="chain")
    def assess_transaction(self,
                          transaction: Dict[str, Any],
                          ml_output: Dict[str, Any],
                          tool_results: Dict[str, Any]) -> FraudAssessment:
        """
        Run all 4 agents and produce final assessment
        
        Args:
            transaction: Transaction details (Amount, card1, DeviceInfo, etc.)
            ml_output: ML model output (fraud_score, shap_values, etc.)
            tool_results: Results from investigation tools
        
        Returns:
            FraudAssessment with complete reasoning
        """
        
        import time
        start_time = time.time()

        print(f"Fraud Score datatype in orchestrator: {type(ml_output.get('fraud_score', 'unknown'))}")
        print(f"Fraud Score value in orchestrator: {ml_output.get('fraud_score', 'unknown')}")
        
        # Agent 1: ML Analysis
        ml_analysis = self.ml_analyst.analyze(
            fraud_score=ml_output['fraud_score'],
            shap_values=ml_output['shap_values'],
            feature_importance=ml_output['feature_importance'],
            raw_score=ml_output.get('fraud_score_raw'),
        )
        ml_analysis = validate_ml_analysis(ml_analysis)
        if ml_analysis.get("error"):
            logger.warning("ML analysis produced validation error: %s", ml_analysis.get("error"))
        
        # Agent 2: Investigation
        investigation = self.investigator.investigate(tool_results)
        
        # Agent 3: Risk Assessment
        risk_assessment = self.risk_assessor.assess(ml_analysis, investigation)
        risk_assessment = validate_risk_assessment(risk_assessment)
        
        # Agent 4: Decision Explanation
        decision = risk_assessment.get('recommendation', 'REVIEW')
        explanation_results = self.explainer.explain(decision, risk_assessment, transaction)
        explanation_text = explanation_results.get("explanation") if isinstance(explanation_results, dict) else str(explanation_results)

        execution_time = (time.time() - start_time) * 1000

        from config import CLAUDE_INPUT_COST_PER_TOKEN, CLAUDE_OUTPUT_COST_PER_TOKEN
        agents = [self.ml_analyst, self.investigator, self.risk_assessor, self.explainer]
        total_input  = sum(a.last_usage["input_tokens"]  for a in agents)
        total_output = sum(a.last_usage["output_tokens"] for a in agents)
        agent_cost_usd = (total_input * CLAUDE_INPUT_COST_PER_TOKEN
                          + total_output * CLAUDE_OUTPUT_COST_PER_TOKEN)

        reasoning_details = {
            'ml_analysis': ml_analysis,
            'investigation_findings': investigation,
            'risk_assessment': risk_assessment,
            'decision_explanation': explanation_results
        }

        return FraudAssessment(
            transaction_id=transaction.get('TransactionID', 'unknown'),
            risk_level=risk_assessment.get('risk_level', 'MEDIUM'),
            risk_score=risk_assessment.get('risk_score', 0.5),
            confidence=risk_assessment.get('confidence', 0.5),
            ml_analysis=ml_analysis,
            investigation_findings=investigation,
            risk_assessment=risk_assessment,
            customer_explanation=explanation_text,
            decision=decision,
            reasoning=json.dumps(reasoning_details, indent=2, default=str),
            reasoning_details=reasoning_details,
            execution_time_ms=execution_time,
            agent_cost_usd=agent_cost_usd,
        )

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Fraud Investigation Agent System")
    print("=" * 80)
    print("""
    4 Specialized Agents:
    1. ML Analyst - Interprets model outputs
    2. Transaction Investigator - Gathers evidence
    3. Risk Assessor - Synthesizes decision
    4. Decision Explainer - Creates customer explanation
    
    Usage:
    orchestrator = FraudInvestigationOrchestrator()
    
    assessment = orchestrator.assess_transaction(
        transaction={...},
        ml_output={...},
        tool_results={...}
    )
    
    Output: FraudAssessment with:
    - risk_level: LOW/MEDIUM/HIGH/CRITICAL
    - decision: APPROVE/SOFT_DECLINE/REVIEW/BLOCK
    - customer_explanation: Plain English
    - reasoning: Complete chain-of-thought
    - execution_time_ms: Time taken
    """)