
from agents.tool_execution import ToolExecutor
from agents.tool_selection import AdaptiveToolSelector
from agents.investigator import FraudInvestigationOrchestrator


def investigate_transaction(transaction, all_transactions, fraud_score, mode="adaptive"):
    """Run investigation tools for transaction.

    Args:
        mode: "adaptive" — AdaptiveToolSelector picks tools based on ML score
                           and latency budget; skips all tools when confident.
              "all"      — Always run every tool regardless of score.
    """
    print(f"\n[INVESTIGATION TOOLS]  mode={mode}")

    tool_executor = ToolExecutor(all_transactions)

    if mode == "adaptive":
        selector = AdaptiveToolSelector()
        selection = selector.select_tools(fraud_score)
        selected_tools = selection['tools']
        print(f"  Selector: {selection['reason']}")

        if not selected_tools:
            print(f"  Skipping all tools (confident enough from ML score alone)")
            return {}

        print(f"  Running: {selected_tools}  (~{selection.get('estimated_latency_ms', '?')}ms)")
        tool_results_raw = tool_executor.execute_selected(transaction, selected_tools)
    else:
        print(f"  Running all tools")
        tool_results_raw = tool_executor.execute_all(transaction)

    # Convert ToolResult objects to dicts for agent
    tool_results = {}
    for tool_name, result in tool_results_raw.items():
        tool_results[tool_name] = {
            'result': result.result,
            'confidence': result.confidence,
            'data_points': result.data_points,
            'error': result.error
        }

        status = "✓" if result.is_successful() else "✗"
        print(f"  {status} {tool_name}: confidence={result.confidence:.2f}, n={result.data_points}")

    return tool_results


 
def run_agents(transaction, ml_output, tool_results):
    """Run all 4 agents for complete assessment"""
    
    print(f"\n[AGENT REASONING]")
    
    orchestrator = FraudInvestigationOrchestrator()
    
    assessment = orchestrator.assess_transaction(
        transaction=transaction,
        ml_output=ml_output,
        tool_results=tool_results
    )
    
    print(f"  Risk Level: {assessment.risk_level}")
    print(f"  Risk Score: {assessment.risk_score:.2%}")
    print(f"  Confidence: {assessment.confidence:.0%}")
    print(f"  Decision: {assessment.decision}")
    print(f"  Execution Time: {assessment.execution_time_ms:.1f}ms")
    
    return assessment