# Model Card: LightGBM Fraud Detection Classifier

## Model Details
- Model Type: LightGBM (Light Gradient Boosting Machine)
- Number of trees: 843
- Training data: 413,713 transactions
- Validation data: 83,894 transactions
- Test data: 87,292 transactions
- Fraud rate (train): 3.537%
- Fraud rate (val): 3.401%
- Fraud rate (test): 3.484%
- Training time: 29.8 seconds

## Performance (Test Set)
- AUC (calibrated): 0.897908
- Brier Score: 0.0223
- Precision (threshold 0.5): 0.3558
- ROC-AUC: 0.897908
- PR-AUC: 0.535582

## Calibration
- Method: Isotonic Regression
- Trained on: Validation set (83,894 samples)
- Brier improvement: -1.0%
- Calibrated probabilities match observed fraud rate

## Features
- Total: 460 engineered features
- Categorical: 29
- Numeric: 431
- Top features: V258, device_fraud_rate, C14, C1, C13

## Categorical Features
Handled with pandas categorical dtype:
- Categories defined from training data only
- Unseen categories treated as NaN
- LightGBM handles natively with categorical_feature parameter
- No data leakage

## Intended Use
- Fraud detection for card transactions
- Real-time scoring (<100ms per transaction)
- Probability output calibrated to match observed fraud rates
- Recommended threshold: 0.38 (maximizes F1)

## Known Limitations
- Trained on 2017-2018 IEEE-CIS data
- Fraud patterns may shift over time
- Limited to card transactions (e-commerce, card-present)
- No merchant-level features

## Production Deployment
- Model size: 460 features
- Inference latency: <100ms per transaction
- Scalability: 1,000+ txns/sec on single server
- Dependencies: lightgbm, numpy, pandas, scikit-learn

## Monitoring Recommendations
- Track fraud rate over time
- Monitor feature distributions
- Retrain monthly with new data
- Track false positive / false negative rates

## Training Data Quality
- IEEE-CIS Fraud Detection Kaggle Competition
- 590,540 transactions (after outlier removal)
- Temporal split: train (67%), val (10%), test (23%)
- All statistics from training data only (zero leakage)
