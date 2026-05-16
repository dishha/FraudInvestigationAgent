import pickle
import pandas as pd
from agents.investigator import FraudInvestigationOrchestrator
from agents.tool_execution import ToolExecutor


# Load trained model
with open('model/lightgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/calibrator.pkl', 'rb') as f:
    calibrator = pickle.load(f)

with open('model/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load data
with open('data/features_test.pkl', 'rb') as f:
    X_test, y_test, _ = pickle.load(f)

# Load all transactions for tool context
with open('data/ieee_prepared.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    all_transactions = pd.concat([
        data_dict['train'], data_dict['val'], data_dict['test']
    ])

# Initialize tools and agents
tool_executor = ToolExecutor(all_transactions)