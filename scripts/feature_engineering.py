def transform_features_for_scoring(X_test, feature_names):
    """Ensure features match training format"""
    X_test = X_test[feature_names].copy()  # Select right columns
    X_test = X_test.fillna(0)              # Fill NaN
    return X_test