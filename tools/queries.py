"""
FRAUD INVESTIGATION TOOLS
IEEE-CIS Dataset Implementation

Tools used by Transaction Investigator agent to gather evidence.
Each tool queries historical transaction data and returns structured findings.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import pickle

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ToolResult:
    """
    Standard result format for all tools
    Provides confidence and error handling
    """
    result: Dict[str, Any]
    confidence: float  # 0.0-1.0, how confident is this data?
    data_points: int   # How many samples used?
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def is_successful(self) -> bool:
        return self.error is None
    
    def __repr__(self):
        status = "✓" if self.is_successful() else "✗"
        return f"{status} ToolResult(confidence={self.confidence:.2f}, n={self.data_points}, time={self.execution_time_ms:.1f}ms)"

# ============================================================================
# TOOL 1: CARD VELOCITY CHECKER
# ============================================================================

class CardVelocityChecker:
    """
    Check recent transaction velocity on a card.
    Detects card testing (multiple small txns quickly) and rapid spending.
    """
    
    def __init__(self, all_transactions_df: pd.DataFrame):
        """
        Args:
            all_transactions_df: Complete transaction history (train + val + test)
        """
        self.df = all_transactions_df
    
    def check(self, card_id: int, current_txn_time: int, 
              time_windows: Dict[str, int] = None) -> ToolResult:
        """
        Check velocity on this card within different time windows.
        
        Args:
            card_id: Card identifier (card1 from transaction)
            current_txn_time: Current transaction timestamp (TransactionDT)
            time_windows: {name: seconds} e.g., {'1h': 3600, '24h': 86400}
        
        Returns:
            ToolResult with velocity metrics
        """
        
        if time_windows is None:
            time_windows = {'1m': 60, '5m': 300, '1h': 3600, '24h': 86400}
        
        try:
            # Get card history BEFORE this transaction
            card_history = self.df[
                (self.df['card1'] == card_id) &
                (self.df['TransactionDT'] < current_txn_time)
            ].sort_values('TransactionDT', ascending=False)
            
            if len(card_history) == 0:
                # New card
                return ToolResult(
                    result={
                        'card_id': card_id,
                        'is_new_card': True,
                        'velocity_by_window': {},
                        'summary': 'New card, no transaction history'
                    },
                    confidence=0.95,  # High confidence (fact)
                    data_points=0
                )
            
            # Calculate velocity in each window
            velocities = {}
            amounts_by_window = {}
            countries_by_window = {}
            min_interval = {}
            
            for window_name, window_seconds in time_windows.items():
                window_start = current_txn_time - window_seconds
                window_txns = card_history[card_history['TransactionDT'] > window_start]
                
                count = len(window_txns)
                velocities[window_name] = count
                amounts_by_window[window_name] = window_txns['TransactionAmt'].tolist() if len(window_txns) > 0 else []
                
                # Get countries
                if 'addr2' in window_txns.columns:
                    countries_by_window[window_name] = window_txns['addr2'].unique().tolist()
                else:
                    countries_by_window[window_name] = []
                
                # Minimum time between transactions (in seconds)
                if len(window_txns) > 1:
                    time_diffs = window_txns['TransactionDT'].diff().abs()
                    min_interval[window_name] = int(time_diffs.min())
                else:
                    min_interval[window_name] = None
            
            # Card statistics
            card_avg_amount = card_history['TransactionAmt'].mean()
            card_max_amount = card_history['TransactionAmt'].max()
            card_num_countries = card_history['addr2'].nunique() if 'addr2' in card_history.columns else 0
            days_since_first_txn = (current_txn_time - card_history['TransactionDT'].min()) / 86400
            
            # Fraud risk assessment
            fraud_risk_signals = []
            
            # Card testing pattern: 3+ txns in 1 hour with small amounts
            if velocities['1h'] >= 3:
                small_txns = [a for a in amounts_by_window['1h'] if a < 50]
                if len(small_txns) >= 2:
                    fraud_risk_signals.append({
                        'signal': 'card_testing',
                        'description': f'{len(small_txns)} small txns (<$50) in 1 hour',
                        'severity': 'high'
                    })
            
            # Rapid spending: 5+ in 1 hour
            if velocities['1h'] >= 5:
                fraud_risk_signals.append({
                    'signal': 'rapid_spending',
                    'description': f'{velocities["1h"]} transactions in 1 hour',
                    'severity': 'medium'
                })
            
            # Multiple countries in 24h
            if len(countries_by_window['24h']) > 2:
                fraud_risk_signals.append({
                    'signal': 'multi_country_24h',
                    'description': f'{len(countries_by_window["24h"])} different countries in 24h',
                    'severity': 'medium'
                })
            
            return ToolResult(
                result={
                    'card_id': card_id,
                    'is_new_card': False,
                    'velocity_by_window': velocities,
                    'amounts_by_window': amounts_by_window,
                    'countries_by_window': countries_by_window,
                    'min_interval_seconds': min_interval,
                    'card_stats': {
                        'avg_amount': float(card_avg_amount),
                        'max_amount': float(card_max_amount),
                        'num_countries': int(card_num_countries),
                        'days_active': float(days_since_first_txn),
                        'total_txns': len(card_history)
                    },
                    'fraud_risk_signals': fraud_risk_signals,
                    'summary': f'{velocities["1h"]} txns in 1h, {velocities["24h"]} in 24h'
                },
                confidence=0.95,  # High confidence (factual data)
                data_points=len(card_history)
            )
        
        except Exception as e:
            return ToolResult(
                result={'error': str(e)},
                confidence=0.0,
                data_points=0,
                error=f"Card velocity check failed: {str(e)}"
            )

# ============================================================================
# TOOL 2: DEVICE PROFILE CHECKER
# ============================================================================

class DeviceProfileChecker:
    """
    Check device reputation and transaction history.
    Detects if device is new, how many cards used it, fraud rate.
    """
    
    def __init__(self, all_transactions_df: pd.DataFrame):
        self.df = all_transactions_df
    
    def check(self, device_id: str, current_txn_time: int) -> ToolResult:
        """
        Check device profile and history.
        
        Args:
            device_id: DeviceInfo from transaction
            current_txn_time: Current transaction timestamp
        
        Returns:
            ToolResult with device metrics
        """
        
        try:
            if pd.isna(device_id) or device_id == '':
                return ToolResult(
                    result={
                        'device_id': None,
                        'note': 'DeviceInfo not present on transaction — no signal available'
                    },
                    confidence=0.0,
                    data_points=0,
                    error="Device ID missing"
                )
            
            # Get device history
            device_history = self.df[
                (self.df['DeviceInfo'] == device_id) &
                (self.df['TransactionDT'] < current_txn_time)
            ]
            
            if len(device_history) == 0:
                # New device
                return ToolResult(
                    result={
                        'device_id': device_id,
                        'is_new_device': True,
                        'device_age_days': 0,
                        'num_cards': 0,
                        'fraud_rate': None,
                        'fraud_risk_signals': [
                            {
                                'signal': 'new_device',
                                'description': 'First transaction from this device',
                                'severity': 'medium'
                            }
                        ],
                        'summary': 'New device, never seen before'
                    },
                    confidence=0.95,
                    data_points=0
                )
            
            # Device statistics
            num_cards = device_history['card1'].nunique()
            device_age_days = (current_txn_time - device_history['TransactionDT'].min()) / 86400
            device_fraud_count = device_history['isFraud'].sum()
            device_fraud_rate = device_history['isFraud'].mean()
            
            # Device risk signals
            fraud_risk_signals = []
            
            # High fraud rate from device
            if device_fraud_rate > 0.1:  # >10% fraud
                fraud_risk_signals.append({
                    'signal': 'high_device_fraud_rate',
                    'description': f'{device_fraud_rate:.1%} of txns from device are fraud',
                    'severity': 'high'
                })
            
            # Multiple cards on device
            if num_cards > 3:
                fraud_risk_signals.append({
                    'signal': 'multiple_cards_on_device',
                    'description': f'{num_cards} different cards used on this device',
                    'severity': 'medium'
                })
            
            # Very new device (used for <1 hour)
            if device_age_days < 1/24:  # Less than 1 hour
                fraud_risk_signals.append({
                    'signal': 'very_new_device',
                    'description': f'Device created < 1 hour ago',
                    'severity': 'medium'
                })
            
            return ToolResult(
                result={
                    'device_id': device_id,
                    'is_new_device': False,
                    'device_age_days': float(device_age_days),
                    'num_cards': int(num_cards),
                    'num_transactions': len(device_history),
                    'fraud_rate': float(device_fraud_rate),
                    'fraud_count': int(device_fraud_count),
                    'avg_amount': float(device_history['TransactionAmt'].mean()),
                    'fraud_risk_signals': fraud_risk_signals,
                    'summary': f'{num_cards} cards, {device_fraud_rate:.1%} fraud rate'
                },
                confidence=0.90,  # High confidence (factual)
                data_points=len(device_history)
            )
        
        except Exception as e:
            return ToolResult(
                result={'error': str(e)},
                confidence=0.0,
                data_points=0,
                error=f"Device check failed: {str(e)}"
            )

# ============================================================================
# TOOL 3: EMAIL DOMAIN CHECKER
# ============================================================================

class EmailDomainChecker:
    """
    Check email domain reputation.
    Detects if domain is free email (trusted) or suspicious.
    """
    
    # Known high-risk domains (from fraud literature)
    HIGH_RISK_DOMAINS = {
        'mailinator.com', 'tempmail.com', 'throwaway.email',
        'guerrillamail.com', 'maildrop.cc', '10minutemail.com',
        # Add more as discovered
    }
    
    # Known trusted domains
    TRUSTED_DOMAINS = {
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'aol.com', 'mail.com', 'protonmail.com',
        # Company domains are generally trusted
    }
    
    def __init__(self, all_transactions_df: pd.DataFrame):
        self.df = all_transactions_df
    
    def check(self, email_domain: str, current_txn_time: Optional[int] = None) -> ToolResult:
        """
        Check email domain reputation.

        Args:
            email_domain: P_emaildomain from transaction
            current_txn_time: TransactionDT of the transaction being scored.
                Only transactions BEFORE this time are used, preventing future-data leakage.

        Returns:
            ToolResult with domain metrics
        """

        try:
            if pd.isna(email_domain) or email_domain == '':
                return ToolResult(
                    result={
                        'domain': None,
                        'is_risky': None,
                        'note': 'P_emaildomain not present on transaction — no signal available'
                    },
                    confidence=0.0,
                    data_points=0,
                    error="Email domain missing"
                )

            # Check domain reputation
            if email_domain.lower() in self.HIGH_RISK_DOMAINS:
                return ToolResult(
                    result={
                        'domain': email_domain,
                        'is_risky': True,
                        'domain_type': 'temporary_email',
                        'fraud_risk_signals': [
                            {
                                'signal': 'temporary_email_domain',
                                'description': f'Known temporary email service: {email_domain}',
                                'severity': 'high'
                            }
                        ],
                        'summary': 'Temporary email domain (high risk)'
                    },
                    confidence=0.95,
                    data_points=0
                )

            # Get domain statistics from transaction history — only prior transactions.
            domain_txns = self.df[self.df['P_emaildomain'] == email_domain]
            if current_txn_time is not None:
                domain_txns = domain_txns[domain_txns['TransactionDT'] < current_txn_time]
            
            if len(domain_txns) == 0:
                return ToolResult(
                    result={
                        'domain': email_domain,
                        'is_risky': False,
                        'domain_type': 'new_domain',
                        'fraud_rate': None,
                        'fraud_risk_signals': [
                            {
                                'signal': 'new_email_domain',
                                'description': 'First transaction from this email domain',
                                'severity': 'low'
                            }
                        ],
                        'summary': 'New email domain (no history)'
                    },
                    confidence=0.80,
                    data_points=0
                )
            
            # Domain statistics
            domain_fraud_rate = domain_txns['isFraud'].mean()
            domain_type = 'trusted' if email_domain.lower() in self.TRUSTED_DOMAINS else 'unknown'
            
            fraud_risk_signals = []
            if domain_fraud_rate > 0.08:  # >8% fraud
                fraud_risk_signals.append({
                    'signal': 'high_domain_fraud_rate',
                    'description': f'{domain_fraud_rate:.1%} of txns from domain are fraud',
                    'severity': 'medium'
                })
            
            return ToolResult(
                result={
                    'domain': email_domain,
                    'is_risky': domain_fraud_rate > 0.08,
                    'domain_type': domain_type,
                    'fraud_rate': float(domain_fraud_rate),
                    'transaction_count': len(domain_txns),
                    'fraud_risk_signals': fraud_risk_signals,
                    'summary': f'{domain_type}, {domain_fraud_rate:.1%} fraud rate'
                },
                confidence=0.85,
                data_points=len(domain_txns)
            )
        
        except Exception as e:
            return ToolResult(
                result={'error': str(e)},
                confidence=0.0,
                data_points=0,
                error=f"Email domain check failed: {str(e)}"
            )

# ============================================================================
# TOOL 4: ADDRESS CLUSTER CHECKER
# ============================================================================

class AddressClusterChecker:
    """
    Check address reputation and clustering.
    Detects fraud rings (multiple cards at same address).
    """
    
    def __init__(self, all_transactions_df: pd.DataFrame):
        self.df = all_transactions_df
    
    def check(self, addr1: int, addr2: int, current_txn_time: int) -> ToolResult:
        """
        Check address cluster metrics.
        
        Args:
            addr1: Billing address (addr1 from transaction)
            addr2: Country code (addr2 from transaction)
            current_txn_time: Current transaction timestamp
        
        Returns:
            ToolResult with address metrics
        """
        
        try:
            if pd.isna(addr1) or pd.isna(addr2):
                return ToolResult(
                    result={'error': 'Missing address data'},
                    confidence=0.0,
                    data_points=0,
                    error="Address information is incomplete"
                )
            
            # Get address history
            addr_txns = self.df[
                (self.df['addr1'] == addr1) &
                (self.df['addr2'] == addr2) &
                (self.df['TransactionDT'] < current_txn_time)
            ]
            
            if len(addr_txns) == 0:
                return ToolResult(
                    result={
                        'addr1': addr1,
                        'addr2': addr2,
                        'is_new_address': True,
                        'num_cards': 0,
                        'num_devices': 0,
                        'fraud_risk_signals': [
                            {
                                'signal': 'new_address',
                                'description': 'First transaction from this address',
                                'severity': 'low'
                            }
                        ],
                        'summary': 'New address, no history'
                    },
                    confidence=0.95,
                    data_points=0
                )
            
            # Address statistics
            num_cards = addr_txns['card1'].nunique()
            num_devices = addr_txns['DeviceInfo'].nunique() if 'DeviceInfo' in addr_txns.columns else 0
            addr_fraud_rate = addr_txns['isFraud'].mean()
            addr_fraud_count = addr_txns['isFraud'].sum()
            
            # Fraud ring detection
            fraud_risk_signals = []

            # Require both elevated card count AND elevated fraud rate to call it a fraud ring.
            # High card count alone just means a busy/shared location (apartment, campus, mall).
            if num_cards > 5 and addr_fraud_rate > 0.12:
                fraud_risk_signals.append({
                    'signal': 'fraud_ring_suspect',
                    'description': f'{num_cards} different cards from same address with {addr_fraud_rate:.1%} fraud rate',
                    'severity': 'high'
                })
            elif num_cards > 5:
                fraud_risk_signals.append({
                    'signal': 'high_volume_address',
                    'description': f'{num_cards} cards from same address but only {addr_fraud_rate:.1%} fraud rate — likely shared/commercial location',
                    'severity': 'low'
                })
            elif num_cards > 2:
                fraud_risk_signals.append({
                    'signal': 'multiple_cards_address',
                    'description': f'{num_cards} cards from same address',
                    'severity': 'medium'
                })

            # High fraud rate from address
            if addr_fraud_rate > 0.12:  # >12% fraud
                fraud_risk_signals.append({
                    'signal': 'high_address_fraud_rate',
                    'description': f'{addr_fraud_rate:.1%} of txns from address are fraud',
                    'severity': 'high'
                })
            
            return ToolResult(
                result={
                    'addr1': addr1,
                    'addr2': addr2,
                    'is_new_address': False,
                    'num_cards': int(num_cards),
                    'num_devices': int(num_devices),
                    'num_transactions': len(addr_txns),
                    'fraud_rate': float(addr_fraud_rate),
                    'fraud_count': int(addr_fraud_count),
                    'avg_amount': float(addr_txns['TransactionAmt'].mean()),
                    'fraud_risk_signals': fraud_risk_signals,
                    'summary': f'{num_cards} cards, {addr_fraud_rate:.1%} fraud rate'
                },
                confidence=0.90,
                data_points=len(addr_txns)
            )
        
        except Exception as e:
            return ToolResult(
                result={'error': str(e)},
                confidence=0.0,
                data_points=0,
                error=f"Address check failed: {str(e)}"
            )

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Tools for IEEE-CIS Fraud Investigation")
    print("=" * 80)
    print("""
    Tools implemented:
    1. CardVelocityChecker - Card transaction patterns
    2. DeviceProfileChecker - Device reputation
    3. EmailDomainChecker - Email domain risk
    4. AddressClusterChecker - Address fraud rings
    
    Each tool returns ToolResult with:
    - result: Findings dict
    - confidence: 0.0-1.0
    - data_points: Sample size
    - error: Error message or None
    
    Usage in agents:
    executor = ToolExecutor(all_transactions_df)
    results = executor.execute_all(transaction)
    
    # Access results
    velocity_result = results['velocity']
    if velocity_result.is_successful():
        print(f"Card has {velocity_result.result['velocity_by_window']['1h']} txns in 1h")
    """)