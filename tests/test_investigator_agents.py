import os
import json
import unittest
from types import SimpleNamespace

import pandas as pd

from agents import investigator
from agents.investigator import (
    FraudInvestigationOrchestrator,
    FraudAssessment,
    MLAnalyst,
    TransactionInvestigator,
    RiskAssessor,
    DecisionExplainer,
)
from agents.tool_selection import AdaptiveToolSelector
from agents.tool_execution import ToolExecutor
from tools.queries import ToolResult
from scripts.fraud_investigation_agents import investigate_transaction


class FakeResponse:
    def __init__(self, text):
        self.content = [SimpleNamespace(text=text)]


class FakeClient:
    def __init__(self, response_text):
        self._response_text = response_text

    @property
    def messages(self):
        return self

    def create(self, *args, **kwargs):
        return FakeResponse(self._response_text)


class InvestigatorAgentTests(unittest.TestCase):
    def setUp(self):
        # Ensure org initializes in mock mode when no credentials are present
        self.orig_api = os.environ.pop("ANTHROPIC_API_KEY", None)
        self.orig_token = os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        investigator.MOCK_MODE = False

    def tearDown(self):
        if self.orig_api is not None:
            os.environ["ANTHROPIC_API_KEY"] = self.orig_api
        if self.orig_token is not None:
            os.environ["ANTHROPIC_AUTH_TOKEN"] = self.orig_token
        investigator.MOCK_MODE = False

    def test_orchestrator_falls_back_to_mock_mode_without_api_creds(self):
        orchestrator = FraudInvestigationOrchestrator()
        self.assertIsNone(orchestrator.client)
        self.assertTrue(investigator.MOCK_MODE)

        assessment = orchestrator.assess_transaction(
            transaction={"TransactionID": "tx123", "Amount": 20.0},
            ml_output={
                "fraud_score": 0.32,
                "shap_values": None,
                "feature_importance": [("v1", 0.1), ("v2", -0.05)],
            },
            tool_results={"some_tool": {"result": {}, "confidence": 0.5, "data_points": 0, "error": None}},
        )

        self.assertIsInstance(assessment, FraudAssessment)
        self.assertEqual(assessment.transaction_id, "tx123")
        self.assertIn(assessment.decision, {"APPROVE", "SOFT_DECLINE", "REVIEW", "BLOCK"})
        self.assertIsInstance(assessment.customer_explanation, str)
        self.assertIsInstance(assessment.ml_analysis, dict)
        self.assertIsInstance(assessment.risk_assessment, dict)

    def test_ml_analyst_returns_fallback_on_parse_error(self):
        investigator.MOCK_MODE = False
        fake_client = FakeClient("not a json response")
        analyst = MLAnalyst(fake_client)

        result = analyst.analyze(
            fraud_score=0.42,
            shap_values=None,
            feature_importance=[("v1", "0.1")],
        )

        self.assertEqual(result["fraud_score"], 0.42)
        self.assertEqual(result["confidence"], 0.5)
        self.assertIn("error", result)

    def test_transaction_investigator_returns_fallback_on_parse_error(self):
        investigator.MOCK_MODE = False
        fake_client = FakeClient("not a json response")
        investigator_agent = TransactionInvestigator(fake_client)

        result = investigator_agent.investigate({"tool": {"result": {}, "confidence": 0.7}})

        self.assertEqual(result["fraud_signals"], [])
        self.assertEqual(result["overall_risk"], "MEDIUM")
        self.assertIn("error", result)

    def test_risk_assessor_returns_fallback_on_parse_error(self):
        investigator.MOCK_MODE = False
        fake_client = FakeClient("not a json response")
        assessor = RiskAssessor(fake_client)

        result = assessor.assess(
            ml_analysis={"fraud_score": 0.55},
            investigation={"fraud_signals": []},
        )

        self.assertEqual(result["risk_level"], "MEDIUM")
        self.assertEqual(result["recommendation"], "REVIEW")
        self.assertIn("error", result)

    def test_decision_explainer_returns_mock_text_when_mock_mode_enabled(self):
        investigator.MOCK_MODE = True
        explainer = DecisionExplainer(None)
        explanation = explainer.explain(
            decision="REVIEW",
            risk_assessment={"risk_level": "MEDIUM", "key_signals": ["velocity"]},
            transaction={"Amount": 100.0},
        )

        self.assertIsInstance(explanation, dict)
        self.assertIn("explanation", explanation)
        self.assertIsInstance(explanation["explanation"], str)
        self.assertIn("verify this is you", explanation["explanation"])


# =============================================================================
# Helpers shared by tool tests
# =============================================================================

def _make_df():
    """Minimal transaction DataFrame that satisfies all four tool checkers."""
    return pd.DataFrame({
        'card1':         [1001, 1001, 1002],
        'TransactionDT': [1000, 2000, 3000],
        'DeviceInfo':    ['device_A', 'device_A', 'device_B'],
        'P_emaildomain': ['gmail.com', 'gmail.com', 'yahoo.com'],
        'addr1':         [100, 100, 200],
        'addr2':         [1, 1, 2],
        'Amount':        [50.0, 75.0, 100.0],
        'isFraud':       [0, 0, 1],
    })


def _make_transaction():
    """Transaction that arrives after all rows in _make_df so history exists."""
    return pd.Series({
        'card1':         1001,
        'TransactionDT': 9999,
        'DeviceInfo':    'device_A',
        'P_emaildomain': 'gmail.com',
        'addr1':         100,
        'addr2':         1,
        'Amount':        60.0,
    })


# =============================================================================
# AdaptiveToolSelector tests
# =============================================================================

class AdaptiveToolSelectorTests(unittest.TestCase):

    def setUp(self):
        self.selector = AdaptiveToolSelector()

    def test_high_score_skips_all_tools(self):
        result = self.selector.select_tools(0.9)
        self.assertEqual(result['tools'], [])

    def test_low_score_skips_all_tools(self):
        result = self.selector.select_tools(0.1)
        self.assertEqual(result['tools'], [])

    def test_medium_score_always_includes_velocity_and_email(self):
        for score in [0.25, 0.5, 0.75]:
            with self.subTest(score=score):
                result = self.selector.select_tools(score)
                self.assertIn('velocity', result['tools'])
                self.assertIn('email', result['tools'])

    def test_score_above_0_4_includes_address(self):
        result = self.selector.select_tools(0.5)
        self.assertIn('address', result['tools'])

    def test_score_at_or_below_0_4_excludes_address(self):
        result = self.selector.select_tools(0.3)
        self.assertNotIn('address', result['tools'])

    def test_estimated_latency_present_for_medium_score(self):
        result = self.selector.select_tools(0.5)
        self.assertIn('estimated_latency_ms', result)
        self.assertIsInstance(result['estimated_latency_ms'], (int, float))

    def test_reason_string_present(self):
        for score in [0.1, 0.5, 0.9]:
            with self.subTest(score=score):
                result = self.selector.select_tools(score)
                self.assertIn('reason', result)
                self.assertIsInstance(result['reason'], str)


# =============================================================================
# ToolExecutor.execute_selected tests
# =============================================================================

class ToolExecutorSelectTests(unittest.TestCase):

    def setUp(self):
        self.executor = ToolExecutor(_make_df())
        self.transaction = _make_transaction()

    def test_only_requested_tools_are_run(self):
        results = self.executor.execute_selected(self.transaction, ['velocity', 'email'])
        self.assertIn('velocity', results)
        self.assertIn('email', results)
        self.assertNotIn('device', results)
        self.assertNotIn('address', results)

    def test_unknown_tool_name_is_silently_skipped(self):
        results = self.executor.execute_selected(self.transaction, ['velocity', 'nonexistent'])
        self.assertIn('velocity', results)
        self.assertNotIn('nonexistent', results)

    def test_empty_tool_list_returns_empty_dict(self):
        results = self.executor.execute_selected(self.transaction, [])
        self.assertEqual(results, {})

    def test_returns_tool_result_instances(self):
        results = self.executor.execute_selected(self.transaction, ['velocity', 'device'])
        for name, result in results.items():
            with self.subTest(tool=name):
                self.assertIsInstance(result, ToolResult)

    def test_all_four_tools_can_be_selected(self):
        results = self.executor.execute_selected(
            self.transaction, ['velocity', 'email', 'device', 'address']
        )
        self.assertEqual(set(results.keys()), {'velocity', 'email', 'device', 'address'})


# =============================================================================
# investigate_transaction mode tests
# =============================================================================

class InvestigateTransactionModeTests(unittest.TestCase):

    def test_adaptive_high_score_returns_empty_dict(self):
        result = investigate_transaction(
            _make_transaction(), _make_df(), fraud_score=0.95, mode="adaptive"
        )
        self.assertEqual(result, {})

    def test_adaptive_low_score_returns_empty_dict(self):
        result = investigate_transaction(
            _make_transaction(), _make_df(), fraud_score=0.05, mode="adaptive"
        )
        self.assertEqual(result, {})

    def test_adaptive_medium_score_returns_tool_results(self):
        result = investigate_transaction(
            _make_transaction(), _make_df(), fraud_score=0.5, mode="adaptive"
        )
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        self.assertIn('velocity', result)

    def test_adaptive_medium_above_0_4_includes_address(self):
        result = investigate_transaction(
            _make_transaction(), _make_df(), fraud_score=0.55, mode="adaptive"
        )
        self.assertIn('address', result)

    def test_adaptive_medium_below_0_4_excludes_address(self):
        result = investigate_transaction(
            _make_transaction(), _make_df(), fraud_score=0.3, mode="adaptive"
        )
        self.assertNotIn('address', result)

    def test_all_mode_runs_every_tool(self):
        result = investigate_transaction(
            _make_transaction(), _make_df(), fraud_score=0.5, mode="all"
        )
        self.assertEqual(set(result.keys()), {'velocity', 'email', 'device', 'address'})

    def test_all_mode_ignores_score(self):
        # Even a high-confidence score should not gate tools in "all" mode
        result = investigate_transaction(
            _make_transaction(), _make_df(), fraud_score=0.99, mode="all"
        )
        self.assertIn('velocity', result)

    def test_each_result_has_expected_keys(self):
        result = investigate_transaction(
            _make_transaction(), _make_df(), fraud_score=0.5, mode="adaptive"
        )
        for tool_name, tool_result in result.items():
            with self.subTest(tool=tool_name):
                self.assertIn('result', tool_result)
                self.assertIn('confidence', tool_result)
                self.assertIn('data_points', tool_result)
                self.assertIn('error', tool_result)


if __name__ == "__main__":
    unittest.main()
