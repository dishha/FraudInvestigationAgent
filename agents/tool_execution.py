from typing import Dict, List
import pandas as pd
from tools.queries import CardVelocityChecker, DeviceProfileChecker, EmailDomainChecker, AddressClusterChecker
from tools.queries import ToolResult

class ToolExecutor:
    """
    Manages all tools with timeout and error handling
    """
    
    def __init__(self, all_transactions_df: pd.DataFrame, timeout_seconds: float = 2.0):
        self.all_txns = all_transactions_df
        self.timeout = timeout_seconds
        
        # Initialize all tools
        self.velocity_checker = CardVelocityChecker(all_transactions_df)
        self.device_checker = DeviceProfileChecker(all_transactions_df)
        self.email_checker = EmailDomainChecker(all_transactions_df)
        self.address_checker = AddressClusterChecker(all_transactions_df)
    
    def execute_all(self, transaction: pd.Series) -> Dict[str, ToolResult]:
        """
        Execute all tools for a transaction with error handling
        """
        results = {}
        
        # Tool 1: Card Velocity
        try:
            results['velocity'] = self.velocity_checker.check(
                card_id=transaction['card1'],
                current_txn_time=transaction['TransactionDT']
            )
        except Exception as e:
            results['velocity'] = ToolResult(
                result={}, confidence=0.0, data_points=0,
                error=f"Velocity check timeout/error: {str(e)}"
            )
        
        # Tool 2: Device Profile
        try:
            results['device'] = self.device_checker.check(
                device_id=transaction.get('DeviceInfo'),
                current_txn_time=transaction['TransactionDT']
            )
        except Exception as e:
            results['device'] = ToolResult(
                result={}, confidence=0.0, data_points=0,
                error=f"Device check timeout/error: {str(e)}"
            )
        
        # Tool 3: Email Domain
        try:
            results['email'] = self.email_checker.check(
                email_domain=transaction.get('P_emaildomain'),
                current_txn_time=transaction['TransactionDT'],
            )
        except Exception as e:
            results['email'] = ToolResult(
                result={}, confidence=0.0, data_points=0,
                error=f"Email check timeout/error: {str(e)}"
            )
        
        # Tool 4: Address Cluster
        try:
            results['address'] = self.address_checker.check(
                addr1=transaction.get('addr1'),
                addr2=transaction.get('addr2'),
                current_txn_time=transaction['TransactionDT']
            )
        except Exception as e:
            results['address'] = ToolResult(
                result={}, confidence=0.0, data_points=0,
                error=f"Address check timeout/error: {str(e)}"
            )
        
        return results

    def execute_selected(self, transaction: pd.Series, tool_names: List[str]) -> Dict[str, ToolResult]:
        """
        Execute only the specified tools. Called by AdaptiveToolSelector output.
        """
        dispatch = {
            'velocity': lambda: self.velocity_checker.check(
                card_id=transaction['card1'],
                current_txn_time=transaction['TransactionDT']
            ),
            'email': lambda: self.email_checker.check(
                email_domain=transaction.get('P_emaildomain'),
                current_txn_time=transaction['TransactionDT'],
            ),
            'device': lambda: self.device_checker.check(
                device_id=transaction.get('DeviceInfo'),
                current_txn_time=transaction['TransactionDT']
            ),
            'address': lambda: self.address_checker.check(
                addr1=transaction.get('addr1'),
                addr2=transaction.get('addr2'),
                current_txn_time=transaction['TransactionDT']
            ),
        }

        results = {}
        for name in tool_names:
            if name not in dispatch:
                continue
            try:
                results[name] = dispatch[name]()
            except Exception as e:
                results[name] = ToolResult(
                    result={}, confidence=0.0, data_points=0,
                    error=f"{name} check error: {str(e)}"
                )
        return results
