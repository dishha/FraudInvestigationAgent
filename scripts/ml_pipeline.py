import pandas as pd

def score_with_ml(transaction_features, models):
    """Score transaction using trained XGBoost model"""
    
    model = models['model']
    calibrator = models['calibrator']
    feature_names = models['feature_names']
    
    print(f"Transaction Features: {transaction_features}")
    # Raw prediction
    fraud_score_raw = model.predict(transaction_features)[0]
    try:
        fraud_score_raw = float(fraud_score_raw)
    except Exception:
        fraud_score_raw = float(str(fraud_score_raw).strip())
    
    # Calibrated prediction
    fraud_score_calibrated = calibrator.predict([fraud_score_raw])[0]
    try:
        fraud_score_calibrated = float(fraud_score_calibrated)
    except Exception:
        fraud_score_calibrated = float(str(fraud_score_calibrated).strip())

    # Load saved feature importance (< 1ms)
    importance_df = pd.read_csv('model/feature_importance.csv')
    top_features_df = importance_df.head(10)
    try:
        top_features_df['importance'] = top_features_df['importance'].astype(float)
    except Exception:
        top_features_df['importance'] = top_features_df['importance'].apply(lambda x: float(str(x).strip()))
    top_features = list(top_features_df[['feature', 'importance']].itertuples(index=False, name=None))

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transaction_features)
    except ImportError:
        print("SHAP not available")
        
        print(f"\n[ML SCORING]")
        print(f"  Fraud Score (raw): {fraud_score_raw:.4f}")
        print(f"  Fraud Score (calibrated): {fraud_score_calibrated:.4f}")
        print(f"  Top features:")
        for feature_name, importance in top_features:
            try:
                importance = float(importance)
            except Exception:
                importance = float(str(importance).strip())
            print(f"    {feature_name}: {importance:.4f}")
    
    return {
        'fraud_score': fraud_score_calibrated,
        'fraud_score_raw': fraud_score_raw,
        'shap_values': shap_values if 'shap_values' in locals() else None,
        'feature_importance': top_features
    }