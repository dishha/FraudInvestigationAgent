# Model Card: LightGBM Fraud Detection Classifier

## Model Details
- Model Type: LightGBM
- Trees: 843
- Train: 413,713 / Val: 83,894 / Test: 87,292 transactions
- Fraud rate — train: 3.537% / val: 3.401% / test: 3.484%
- Training time: 29.5s

## Performance (Test Set, Calibrated)
- AUC: 0.897908
- ROC-AUC: 0.897908
- PR-AUC: 0.535582
- Brier Score: 0.0223
- Precision @ 0.5: 0.7869
- Recall @ 0.5: 0.3558

## Calibration
- Method: Isotonic Regression on validation set (83,894 samples)
- Brier improvement: -1.0%

## Features
- Total: 460 engineered features
- Top 5: V258, device_fraud_rate, C14, C1, C13

## Intended Use
- Fraud detection for card transactions
- Real-time scoring (<100ms per transaction)
- Recommended threshold: 0.38 (maximizes F1)

## Known Limitations
- Trained on 2017-2018 IEEE-CIS data; fraud patterns may shift
- Within-train target encoding (device_fraud_rate, email_fraud_rate, etc.)
  includes each row's own label — consider out-of-fold encoding before retraining

## Training Data Quality
- IEEE-CIS Kaggle Competition dataset
- Temporal split: train (67%), val (10%), test (23%)
- All group statistics derived from training data only
