# agent/tool_selection.py

class AdaptiveToolSelector:
    """
    Not all tools are equally valuable.
    Not all transactions need all tools.
    Agent selects intelligently.
    """
    
    TOOL_PROPERTIES = {
        'velocity': {'latency_ms': 100, 'informativeness': 0.95, 'priority': 1},
        'email':    {'latency_ms':  50, 'informativeness': 0.55, 'priority': 2},
        'device':   {'latency_ms': 500, 'informativeness': 0.60, 'priority': 3},
        'address':  {'latency_ms': 300, 'informativeness': 0.40, 'priority': 4},
    }
    
    def select_tools(self, fraud_score, target_latency=2000):
        """
        Decision logic:
        - If already confident (score > 0.8 or < 0.2): no tools needed
        - Else: call tools in priority order, stop when confident enough
        - Respect latency budget (don't call 3 slow tools)
        """
        
        if fraud_score > 0.8:
            return {
                'tools': [],
                'reason': 'High fraud score, skip tools'
            }
        
        if fraud_score < 0.2:
            return {
                'tools': [],
                'reason': 'Low fraud score, skip tools'
            }
        
        # Medium uncertainty: call tools in priority order within latency budget
        tools = ['velocity']
        latency = 100

        if latency + 50 < target_latency:
            tools.append('email')
            latency += 50

        if latency + 500 < target_latency:
            tools.append('device')
            latency += 500

        if latency + 300 < target_latency and fraud_score > 0.4:
            tools.append('address')
            latency += 300

        return {
            'tools': tools,
            'reason': f'Medium uncertainty (score {fraud_score:.2f}). '
                     f'Calling tools in priority order.',
            'estimated_latency_ms': latency
        }