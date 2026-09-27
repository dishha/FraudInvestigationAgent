"""
Layer 0: Hard Rules Engine — deterministic, non-overridable fraud checks.

Rules run before ML scoring. A triggered rule makes the decision final;
ML and agents are skipped entirely. The first triggered rule wins.
"""

from dataclasses import dataclass
from typing import Optional, Set

import pandas as pd


@dataclass
class RuleResult:
    triggered: bool
    decision: Optional[str]     # "BLOCK" when triggered, None otherwise
    rule_name: Optional[str]    # e.g. "CARD_VELOCITY_1M"
    evidence: str               # human-readable explanation


class HardRulesEngine:
    """Deterministic guard evaluated before ML scoring and agent investigation.

    Thresholds are class-level constants so they can be overridden in tests
    without touching constructor arguments.
    """

    MAX_TXNS_PER_MINUTE: int = 3
    MAX_TXNS_PER_HOUR: int = 10
    NEW_DEVICE_AMOUNT_LIMIT: float = 500.0   # USD

    def __init__(
        self,
        hist_df: pd.DataFrame,
        blacklisted_devices: Optional[Set[str]] = None,
    ) -> None:
        self.hist_df = hist_df
        self.blacklisted_devices: Set[str] = blacklisted_devices or set()

    # ── Public API ───────────────────────────────────────────────────────────

    def evaluate(self, txn: pd.Series) -> RuleResult:
        """Evaluate all rules in priority order. Returns on the first hit."""
        for rule_fn in (
            self._rule_blacklisted_device,
            self._rule_card_velocity,
            self._rule_large_amount_new_device,
        ):
            result = rule_fn(txn)
            if result.triggered:
                return result
        return RuleResult(
            triggered=False, decision=None, rule_name=None,
            evidence="No hard rules triggered.",
        )

    # ── Rules ────────────────────────────────────────────────────────────────

    def _rule_blacklisted_device(self, txn: pd.Series) -> RuleResult:
        device = txn.get("DeviceInfo")
        if pd.isna(device) or str(device) not in self.blacklisted_devices:
            return self._no_hit()
        return RuleResult(
            triggered=True, decision="BLOCK",
            rule_name="BLACKLISTED_DEVICE",
            evidence=f"Device '{device}' is on the block list.",
        )

    def _rule_card_velocity(self, txn: pd.Series) -> RuleResult:
        card_id = txn.get("card1")
        current_time = txn.get("TransactionDT")
        if pd.isna(card_id) or pd.isna(current_time):
            return self._no_hit()

        prior_times = self.hist_df.loc[
            (self.hist_df["card1"] == card_id)
            & (self.hist_df["TransactionDT"] < current_time),
            "TransactionDT",
        ]

        txns_1m = int((prior_times >= current_time - 60).sum())
        txns_1h = int((prior_times >= current_time - 3600).sum())

        if txns_1m >= self.MAX_TXNS_PER_MINUTE:
            return RuleResult(
                triggered=True, decision="BLOCK",
                rule_name="CARD_VELOCITY_1M",
                evidence=(
                    f"Card {int(card_id)} had {txns_1m} transactions in the last 60 s "
                    f"(limit: {self.MAX_TXNS_PER_MINUTE})."
                ),
            )
        if txns_1h >= self.MAX_TXNS_PER_HOUR:
            return RuleResult(
                triggered=True, decision="BLOCK",
                rule_name="CARD_VELOCITY_1H",
                evidence=(
                    f"Card {int(card_id)} had {txns_1h} transactions in the last hour "
                    f"(limit: {self.MAX_TXNS_PER_HOUR})."
                ),
            )
        return self._no_hit()

    def _rule_large_amount_new_device(self, txn: pd.Series) -> RuleResult:
        device = txn.get("DeviceInfo")
        amount_raw = txn.get("TransactionAmt") or txn.get("Amount")
        current_time = txn.get("TransactionDT")

        if pd.isna(device) or amount_raw is None or pd.isna(current_time):
            return self._no_hit()

        amount = float(amount_raw)
        if amount <= self.NEW_DEVICE_AMOUNT_LIMIT:
            return self._no_hit()

        prior_count = int(
            (
                (self.hist_df["DeviceInfo"] == device)
                & (self.hist_df["TransactionDT"] < current_time)
            ).sum()
        )
        if prior_count == 0:
            return RuleResult(
                triggered=True, decision="BLOCK",
                rule_name="LARGE_AMOUNT_NEW_DEVICE",
                evidence=(
                    f"${amount:.2f} transaction from first-seen device '{device}' "
                    f"(limit: ${self.NEW_DEVICE_AMOUNT_LIMIT:.0f} for new devices)."
                ),
            )
        return self._no_hit()

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _no_hit() -> RuleResult:
        return RuleResult(triggered=False, decision=None, rule_name=None, evidence="")
