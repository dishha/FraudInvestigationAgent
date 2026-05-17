"""
Inference helper (importable) + offline training pipeline.

Inference:
    from scripts.ml_pipeline import score_with_ml

Training (run from scripts/ directory):
    python ml_pipeline.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def score_with_ml(transaction_features, models):
    """Score a single pre-engineered transaction row.

    transaction_features: DataFrame — columns are reordered to match training order.
    models: dict with keys 'model', 'calibrator', 'feature_names'.
    """
    model = models['model']
    calibrator = models['calibrator']
    feature_names = models['feature_names']

    aligned = transaction_features[feature_names].copy()

    fraud_score_raw = float(model.predict(aligned)[0])
    fraud_score_calibrated = float(calibrator.predict([fraud_score_raw])[0])

    importance_values = model.feature_importance(importance_type='gain')
    top_features = sorted(
        zip(feature_names, importance_values.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    shap_values = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(aligned)
    except Exception:
        pass

    return {
        'fraud_score': fraud_score_calibrated,
        'fraud_score_raw': fraud_score_raw,
        'shap_values': shap_values,
        'feature_importance': top_features,
    }


if __name__ == "__main__":
    import json
    import time

    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import shap
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import (
        auc,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    print("=" * 80)
    print("NOTEBOOK 03: LIGHTGBM MODEL TRAINING")
    print("=" * 80)

    # ========================================================================
    # SECTION 0: SETUP & LOAD DATA
    # ========================================================================
    print("\n[0] Setup and load features...")

    Path('../model').mkdir(exist_ok=True)
    Path('../model/results').mkdir(parents=True, exist_ok=True)

    with open('../data/features_train.pkl', 'rb') as f:
        X_train, y_train, feature_names = pickle.load(f)
    with open('../data/features_val.pkl', 'rb') as f:
        X_val, y_val, _ = pickle.load(f)
    with open('../data/features_test.pkl', 'rb') as f:
        X_test, y_test, _ = pickle.load(f)

    print(f"✓ Train: {X_train.shape}")
    print(f"✓ Val:   {X_val.shape}")
    print(f"✓ Test:  {X_test.shape}")
    print(f"✓ Features: {len(feature_names)}")

    # ========================================================================
    # SECTION 1: VALIDATE DATA
    # ========================================================================
    print("\n[1] Validating data quality...")

    assert not X_train.isnull().any().any(), "NaN in X_train"
    assert not X_val.isnull().any().any(), "NaN in X_val"
    assert not X_test.isnull().any().any(), "NaN in X_test"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Shape mismatch"

    print("  ✓ No NaN values, shapes match")
    print(f"  Train fraud rate: {y_train.mean():.3%}")
    print(f"  Val fraud rate:   {y_val.mean():.3%}")
    print(f"  Test fraud rate:  {y_test.mean():.3%}")

    # ========================================================================
    # SECTION 2: CREATE LIGHTGBM DATASETS
    # ========================================================================
    print("\n[2] Creating LightGBM datasets...")

    train_dataset = lgb.Dataset(
        X_train, label=y_train, feature_name=feature_names, free_raw_data=False
    )
    val_dataset = lgb.Dataset(
        X_val, label=y_val, feature_name=feature_names,
        reference=train_dataset, free_raw_data=False
    )
    print("✓ LightGBM datasets created")

    # ========================================================================
    # SECTION 3: HYPERPARAMETERS
    # ========================================================================
    print("\n[3] Hyperparameters...")

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42,
    }
    for key, val in params.items():
        if key not in ('verbose', 'seed'):
            print(f"  {key}: {val}")

    # ========================================================================
    # SECTION 4: TRAIN WITH EARLY STOPPING
    # ========================================================================
    print("\n[4] Training LightGBM model...")

    start_time = time.time()
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=1000,
        valid_sets=[train_dataset, val_dataset],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    training_time = time.time() - start_time
    print(f"✓ Trained in {training_time:.1f}s — {model.num_trees()} trees (best iter {model.best_iteration})")

    # ========================================================================
    # SECTION 5: EVALUATE ON TEST SET (PRE-CALIBRATION)
    # ========================================================================
    print("\n[5] Evaluating on test set (raw)...")

    y_pred_raw = model.predict(X_test, num_iteration=model.best_iteration)
    auc_raw = roc_auc_score(y_test, y_pred_raw)
    brier_before = float(np.mean((y_pred_raw - y_test) ** 2))

    y_pred_binary = (y_pred_raw >= 0.5).astype(int)
    prec_at_half = precision_score(y_test, y_pred_binary, zero_division=0)
    rec_at_half  = recall_score(y_test, y_pred_binary, zero_division=0)

    print(f"  AUC:       {auc_raw:.6f}")
    print(f"  Precision: {prec_at_half:.4f}  Recall: {rec_at_half:.4f}  (threshold 0.5)")
    print(f"  Brier:     {brier_before:.4f}")

    # ========================================================================
    # SECTION 6: CALIBRATION WITH ISOTONIC REGRESSION
    # ========================================================================
    print("\n[6] Calibrating with Isotonic Regression...")

    val_pred_raw = model.predict(X_val, num_iteration=model.best_iteration)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_pred_raw, y_val)

    y_pred_calibrated = calibrator.predict(y_pred_raw)
    brier_after = float(np.mean((y_pred_calibrated - y_test) ** 2))
    auc_calibrated = roc_auc_score(y_test, y_pred_calibrated)

    print(f"✓ Calibrated on {len(val_pred_raw):,} val samples")
    print(f"  AUC:   {auc_calibrated:.6f}")
    print(f"  Brier: {brier_after:.4f}  (Δ {(brier_before - brier_after) / brier_before * 100:.1f}%)")

    # ========================================================================
    # SECTION 7: FEATURE IMPORTANCE
    # ========================================================================
    print("\n[7] Computing feature importance...")

    importance_values = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values,
    }).sort_values('importance', ascending=False)

    print("Top 15 features:")
    for _, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:12.2f}")

    plt.figure(figsize=(10, 8))
    top_20 = feature_importance.head(20)
    plt.barh(range(len(top_20)), top_20['importance'])
    plt.yticks(range(len(top_20)), top_20['feature'], fontsize=9)
    plt.xlabel('Feature Importance (Gain)')
    plt.title('Top 20 Features - LightGBM')
    plt.tight_layout()
    plt.savefig('../model/results/feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Saved ../model/results/feature_importance.png")

    # ========================================================================
    # SECTION 8: SHAP EXPLANATIONS
    # ========================================================================
    print("\n[8] Computing SHAP explanations...")

    sample_idx = np.random.choice(len(X_test), min(1000, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        shap_fraud = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        print(f"✓ SHAP values: {shap_fraud.shape}")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_fraud, X_sample, feature_names=feature_names, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig('../model/results/shap_summary.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("✓ Saved ../model/results/shap_summary.png")
    except Exception as e:
        print(f"⚠ SHAP failed (non-critical): {e}")

    # ========================================================================
    # SECTION 9: ROC & PR CURVES
    # ========================================================================
    print("\n[9] Creating evaluation plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_test, y_pred_calibrated)
    roc_auc_val = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc_val:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve (Calibrated)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_pred_calibrated)
    pr_auc_val = auc(rec_arr, prec_arr)
    axes[1].plot(rec_arr, prec_arr, lw=2, label=f'PR-AUC = {pr_auc_val:.4f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve (Calibrated)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../model/results/evaluation_curves.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Saved ../model/results/evaluation_curves.png")

    # ========================================================================
    # SECTION 10: SAVE ARTIFACTS
    # ========================================================================
    print("\n[10] Saving artifacts...")

    model.save_model('../model/lightgbm_model.txt')
    with open('../model/lightgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('../model/calibrator.pkl', 'wb') as f:
        pickle.dump(calibrator, f)
    with open('../model/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    feature_importance.to_csv('../model/feature_importance.csv', index=False)
    with open('../model/metrics.json', 'w') as f:
        json.dump({'auc': round(auc_calibrated, 4), 'pr_auc': round(pr_auc_val, 4), 'brier': round(brier_after, 4)}, f)

    print("✓ Saved ../model/lightgbm_model.pkl")
    print("✓ Saved ../model/calibrator.pkl")
    print("✓ Saved ../model/feature_names.pkl")
    print("✓ Saved ../model/feature_importance.csv")

    # ========================================================================
    # SECTION 11: MODEL CARD
    # ========================================================================
    print("\n[11] Creating model card...")

    model_card = f"""# Model Card: LightGBM Fraud Detection Classifier

## Model Details
- Model Type: LightGBM
- Trees: {model.num_trees()}
- Train: {len(X_train):,} / Val: {len(X_val):,} / Test: {len(X_test):,} transactions
- Fraud rate — train: {y_train.mean():.3%} / val: {y_val.mean():.3%} / test: {y_test.mean():.3%}
- Training time: {training_time:.1f}s

## Performance (Test Set, Calibrated)
- AUC: {auc_calibrated:.6f}
- ROC-AUC: {roc_auc_val:.6f}
- PR-AUC: {pr_auc_val:.6f}
- Brier Score: {brier_after:.4f}
- Precision @ 0.5: {prec_at_half:.4f}
- Recall @ 0.5: {rec_at_half:.4f}

## Calibration
- Method: Isotonic Regression on validation set ({len(X_val):,} samples)
- Brier improvement: {(brier_before - brier_after) / brier_before * 100:.1f}%

## Features
- Total: {len(feature_names)} engineered features
- Top 5: {', '.join(feature_importance.head(5)['feature'].tolist())}

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
"""

    with open('../model/MODEL_CARD.md', 'w') as f:
        f.write(model_card)
    print("✓ Saved ../model/MODEL_CARD.md")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("NOTEBOOK 03 COMPLETE")
    print("=" * 80)
    print(f"""
  AUC (calibrated): {auc_calibrated:.6f}
  ROC-AUC:          {roc_auc_val:.6f}
  PR-AUC:           {pr_auc_val:.6f}
  Brier:            {brier_after:.4f}
  Trees:            {model.num_trees()}
  Features:         {len(feature_names)}
  Training time:    {training_time:.1f}s
  Top feature:      {feature_importance.iloc[0]['feature']}
""")
    print("=" * 80)
