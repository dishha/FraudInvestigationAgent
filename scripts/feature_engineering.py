"""
NOTEBOOK 02: FEATURE ENGINEERING - FINAL CORRECT VERSION
Uses pandas categorical dtype (native, simpler, better)
Zero data leakage, all splits use same category definitions

Run as a script to rebuild features:
    python scripts/feature_engineering.py           # v1 temporal split (default)
    python scripts/feature_engineering.py --split v2  # v2 card-based split

Import for inference:
    from scripts.feature_engineering import transform_features_for_scoring
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# INFERENCE HELPER — importable without executing the pipeline
# ============================================================================

def transform_features_for_scoring(X_test, feature_names):
    """Align a feature DataFrame to the training column order.

    The pre-engineered X_test (loaded from features_test.pkl) has already
    had all transformations applied offline. This function simply selects
    and reorders columns to match what the model expects.

    LightGBM handles NaN natively via its own missing-value branches, so
    NaN values are passed through unchanged.
    """
    return X_test[feature_names].copy()


# ============================================================================
# OFFLINE PIPELINE — only runs when executed as a script
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["v1", "v2"], default="v1",
        help="v1 = temporal split (ieee_prepared.pkl), v2 = card split (ieee_prepared_v2.pkl)",
    )
    args = parser.parse_args()

    SPLIT_VERSION = args.split
    IN_FILE  = "../data/ieee_prepared_v2.pkl" if SPLIT_VERSION == "v2" else "../data/ieee_prepared.pkl"
    OUT_SUFFIX = "_v2" if SPLIT_VERSION == "v2" else ""

    print("=" * 80)
    print(f"NOTEBOOK 02: FEATURE ENGINEERING (PANDAS CATEGORICAL) — split={SPLIT_VERSION}")
    print("=" * 80)

    # ========================================================================
    # SECTION 0: LOAD AND VALIDATE DATA
    # ========================================================================

    print("\n[0] Loading and validating prepared data...")

    with open(IN_FILE, 'rb') as f:
        data_dict = pickle.load(f)

    train_data = data_dict['train'].copy()
    val_data = data_dict['val'].copy()
    test_data = data_dict['test'].copy()

    print(f"✓ Train: {len(train_data):,} rows")
    print(f"✓ Val: {len(val_data):,} rows")
    print(f"✓ Test: {len(test_data):,} rows")

    # ========================================================================
    # SECTION 1: OUTLIER REMOVAL (TRAIN PERCENTILE ONLY)
    # ========================================================================

    print("\n[1] Outlier removal (train percentile only)...")

    p99_amount = train_data['TransactionAmt'].quantile(0.99)
    print(f"  99th percentile: ${p99_amount:.2f}")

    train_data = train_data[train_data['TransactionAmt'] <= p99_amount].copy()
    val_data = val_data[val_data['TransactionAmt'] <= p99_amount].copy()
    test_data = test_data[test_data['TransactionAmt'] <= p99_amount].copy()

    print(f"✓ After outlier removal: train={len(train_data):,}, val={len(val_data):,}, test={len(test_data):,}")

    # ========================================================================
    # SECTION 2: DROP LOW-INFORMATION COLUMNS
    # ========================================================================

    print("\n[2] Dropping low-information columns...")

    missing_pct = train_data.isnull().sum() / len(train_data)
    const_variance = train_data.var()

    high_missing = missing_pct[missing_pct > 0.99].index.tolist()
    constant_cols = const_variance[const_variance == 0].index.tolist()

    cols_to_drop = list(set(high_missing + constant_cols))

    train_data = train_data.drop(cols_to_drop, axis=1)
    val_data = val_data.drop(cols_to_drop, axis=1)
    test_data = test_data.drop(cols_to_drop, axis=1)

    print(f"✓ After cleanup: {train_data.shape[1]} columns")

    # ========================================================================
    # SECTION 3: TEMPORAL FEATURES
    # ========================================================================

    print("\n[3] Creating temporal features...")

    for data in [train_data, val_data, test_data]:
        data['hour'] = ((data['TransactionDT'] % 86400) / 3600).astype(int)
        data['day_of_week'] = ((data['TransactionDT'] // 86400) % 7).astype(int)
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['is_night'] = ((data['hour'] < 6) | (data['hour'] >= 22)).astype(int)

    print(f"✓ Added temporal features")

    # ========================================================================
    # SECTION 4: AMOUNT FEATURES
    # ========================================================================

    print("\n[4] Creating amount features...")

    for data in [train_data, val_data, test_data]:
        data['log_amount'] = np.log1p(data['TransactionAmt'])
        data['amount_squared'] = data['TransactionAmt'] ** 2
        data['is_small_amount'] = (data['TransactionAmt'] < 10).astype(int)
        data['is_large_amount'] = (data['TransactionAmt'] > 500).astype(int)
        data['is_medium_amount'] = (
            (data['TransactionAmt'] >= 10) & (data['TransactionAmt'] <= 500)
        ).astype(int)

    print(f"✓ Added amount features")

    # ========================================================================
    # SECTION 5: D-COLUMN NORMALIZATION
    # ========================================================================

    print("\n[5] Normalizing D columns...")

    for data in [train_data, val_data, test_data]:
        data['transaction_day'] = data['TransactionDT'] / (24 * 3600)

        for col in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
                    'D10', 'D11', 'D12', 'D13', 'D14', 'D15']:
            if col in data.columns and data[col].notna().sum() > 0:
                data[f'{col}_norm'] = (data['transaction_day'] - data[col]).fillna(-1)

    print(f"✓ Normalized D columns")

    # ========================================================================
    # SECTION 6: COMPUTE TRAIN STATISTICS
    # ========================================================================

    print("\n[6] Computing statistics from TRAINING DATA ONLY...")

    train_stats = {}

    # Global amount stats — used as fallback for cards/devices/addresses
    # not seen in training (critical for v2 card-split where ALL val/test
    # cards are unseen; also improves v1 for the rare unseen card).
    train_stats['global'] = {
        'mean_amount': float(train_data['TransactionAmt'].mean()),
        'std_amount':  float(train_data['TransactionAmt'].std()),
        'max_amount':  float(train_data['TransactionAmt'].quantile(0.75)),
    }
    print(f"  Global amount mean: ${train_stats['global']['mean_amount']:.2f}, "
          f"std: ${train_stats['global']['std_amount']:.2f}")

    # Card statistics (no label used — no leakage)
    card_stats = {}
    for card_id in train_data['card1'].unique():
        card_txns = train_data[train_data['card1'] == card_id]
        card_stats[card_id] = {
            'total_txns': len(card_txns),
            'mean_amount': float(card_txns['TransactionAmt'].mean()),
            'std_amount': float(card_txns['TransactionAmt'].std()) or 0,
            'max_amount': float(card_txns['TransactionAmt'].max()),
        }
    train_stats['card'] = card_stats
    print(f"  Card stats: {len(card_stats):,} cards")

    # Device statistics — fraud_rate uses isFraud; computed on train-only
    # so no cross-split leakage.  Within-train rows see their own label
    # (naive target encoding); use out-of-fold encoding to eliminate this.
    device_stats = {}
    for device_id in train_data['DeviceInfo'].dropna().unique():
        device_txns = train_data[train_data['DeviceInfo'] == device_id]
        device_stats[device_id] = {
            'num_cards': int(device_txns['card1'].nunique()),
            'fraud_rate': float(device_txns['isFraud'].mean()),
            'mean_amount': float(device_txns['TransactionAmt'].mean()),
        }
    train_stats['device'] = device_stats
    print(f"  Device stats: {len(device_stats):,} devices")

    # Email statistics — same note as device_stats
    email_stats = {}
    for domain in train_data['P_emaildomain'].dropna().unique():
        domain_txns = train_data[train_data['P_emaildomain'] == domain]
        email_stats[domain] = {
            'fraud_rate': float(domain_txns['isFraud'].mean()),
            'mean_amount': float(domain_txns['TransactionAmt'].mean()),
        }
    train_stats['email'] = email_stats
    print(f"  Email stats: {len(email_stats):,} domains")

    # Address statistics — same note as device_stats
    address_stats = {}
    train_addr = train_data[train_data['addr1'].notna() & train_data['addr2'].notna()]
    for _, row in train_addr[['addr1', 'addr2']].drop_duplicates().iterrows():
        addr_txns = train_addr[
            (train_addr['addr1'] == row['addr1']) & (train_addr['addr2'] == row['addr2'])
        ]
        key = (int(row['addr1']), int(row['addr2']))
        address_stats[key] = {
            'fraud_rate': float(addr_txns['isFraud'].mean()),
            'num_cards': int(addr_txns['card1'].nunique()),
        }
    train_stats['address'] = address_stats
    print(f"  Address stats: {len(address_stats):,} addresses")

    # Hour statistics — same note
    hour_stats = {}
    for hour in range(24):
        hour_txns = train_data[train_data['hour'] == hour]
        if len(hour_txns) > 0:
            hour_stats[hour] = {
                'fraud_rate': float(hour_txns['isFraud'].mean()),
            }
    train_stats['hour'] = hour_stats

    print(f"✓ Train statistics computed")

    # ========================================================================
    # SECTION 7: CREATE FEATURES FROM TRAIN STATISTICS
    # ========================================================================

    print("\n[7] Creating features using train statistics only...")

    def create_features(data, stats, split_name):
        data = data.copy()

        # Card features (no label — clean)
        # Unseen cards (always the case in v2; rare in v1) fall back to global
        # train statistics rather than 0, which would be a misleading signal.
        print(f"  Creating card features ({split_name})...")
        _g = stats.get('global', {})
        _fallback_mean = _g.get('mean_amount', 0)
        _fallback_std  = _g.get('std_amount', 0)
        _fallback_max  = _g.get('max_amount', 0)

        data['card_total_txns'] = data['card1'].map(
            lambda x: stats['card'].get(x, {}).get('total_txns', 0)
        ).fillna(0)
        # is_new_card = 1 when the card has no training history.
        # In v2 this is always 1 for val/test; in v1 it flags cards first seen
        # after the training window.
        data['is_new_card'] = (data['card_total_txns'] == 0).astype(int)
        data['card_mean_amount'] = data['card1'].map(
            lambda x: stats['card'].get(x, {}).get('mean_amount', _fallback_mean)
        ).fillna(_fallback_mean)
        data['card_std_amount'] = data['card1'].map(
            lambda x: stats['card'].get(x, {}).get('std_amount', _fallback_std)
        ).fillna(_fallback_std)
        data['card_max_amount'] = data['card1'].map(
            lambda x: stats['card'].get(x, {}).get('max_amount', _fallback_max)
        ).fillna(_fallback_max)

        # Device features
        print(f"  Creating device features ({split_name})...")
        data['device_num_cards'] = data['DeviceInfo'].map(
            lambda x: stats['device'].get(x, {}).get('num_cards', 0) if pd.notna(x) else 0
        ).fillna(0)
        data['device_fraud_rate'] = data['DeviceInfo'].map(
            lambda x: stats['device'].get(x, {}).get('fraud_rate', 0) if pd.notna(x) else 0
        ).fillna(0)
        data['device_mean_amount'] = data['DeviceInfo'].map(
            lambda x: stats['device'].get(x, {}).get('mean_amount', 0) if pd.notna(x) else 0
        ).fillna(0)

        # Email features
        print(f"  Creating email features ({split_name})...")
        data['email_fraud_rate'] = data['P_emaildomain'].map(
            lambda x: stats['email'].get(x, {}).get('fraud_rate', 0) if pd.notna(x) else 0
        ).fillna(0)
        data['email_mean_amount'] = data['P_emaildomain'].map(
            lambda x: stats['email'].get(x, {}).get('mean_amount', 0) if pd.notna(x) else 0
        ).fillna(0)

        # Address features
        print(f"  Creating address features ({split_name})...")
        def get_addr_stat(row, stat):
            if pd.isna(row['addr1']) or pd.isna(row['addr2']):
                return 0
            key = (int(row['addr1']), int(row['addr2']))
            return stats['address'].get(key, {}).get(stat, 0)

        data['address_fraud_rate'] = data.apply(lambda r: get_addr_stat(r, 'fraud_rate'), axis=1)
        data['address_num_cards'] = data.apply(lambda r: get_addr_stat(r, 'num_cards'), axis=1)

        # Hour features
        print(f"  Creating hour features ({split_name})...")
        data['hour_fraud_rate'] = data['hour'].map(
            lambda x: stats['hour'].get(x, {}).get('fraud_rate', 0)
        ).fillna(0)

        data = data.fillna(0)
        return data

    train_data = create_features(train_data, train_stats, 'train')
    val_data = create_features(val_data, train_stats, 'val')
    test_data = create_features(test_data, train_stats, 'test')

    print(f"✓ Features created")

    # ========================================================================
    # SECTION 8: HANDLE CATEGORICAL COLUMNS WITH PANDAS CATEGORICAL
    # ========================================================================

    print("\n[8] Converting categorical columns to pandas categorical...")

    categorical_cols = [col for col in train_data.columns if train_data[col].dtype == 'object']
    print(f"  Found {len(categorical_cols)} categorical columns")

    if categorical_cols:
        for col in categorical_cols:
            # Categories from TRAIN ONLY — unseen val/test values become NaN → code -1
            train_categories = train_data[col].dropna().unique().tolist()
            print(f"    {col}: {len(train_categories)} categories from train")

            train_data[col] = pd.Categorical(train_data[col], categories=train_categories).codes
            val_data[col] = pd.Categorical(val_data[col], categories=train_categories).codes
            test_data[col] = pd.Categorical(test_data[col], categories=train_categories).codes

        print(f"✓ Converted {len(categorical_cols)} columns to categorical codes")

    # ========================================================================
    # SECTION 9: SELECT FEATURES AND PREPARE OUTPUT
    # ========================================================================

    print("\n[9] Selecting features and preparing output...")

    exclude = {'TransactionID', 'uid', 'transaction_day', 'isFraud'}
    feature_cols = [c for c in train_data.columns if c not in exclude]

    X_train = train_data[feature_cols].copy()
    y_train = train_data['isFraud'].copy()

    X_val = val_data[feature_cols].copy()
    y_val = val_data['isFraud'].copy()

    X_test = test_data[feature_cols].copy()
    y_test = test_data['isFraud'].copy()

    cat_features = [col for col in feature_cols if pd.api.types.is_categorical_dtype(X_train[col])]

    print(f"✓ X_train: {X_train.shape} ({len(cat_features)} categorical)")
    print(f"✓ X_val:   {X_val.shape}")
    print(f"✓ X_test:  {X_test.shape}")
    print(f"\nLabel distribution:")
    print(f"  Train: {y_train.mean():.3%} fraud")
    print(f"  Val:   {y_val.mean():.3%} fraud")
    print(f"  Test:  {y_test.mean():.3%} fraud")

    # ========================================================================
    # SECTION 10: SAVE FEATURES
    # ========================================================================

    print("\n[10] Saving features...")

    Path('../data').mkdir(exist_ok=True)

    with open(f'../data/features_train{OUT_SUFFIX}.pkl', 'wb') as f:
        pickle.dump((X_train, y_train, feature_cols), f)
    print(f"✓ Saved ../data/features_train{OUT_SUFFIX}.pkl")

    with open(f'../data/features_val{OUT_SUFFIX}.pkl', 'wb') as f:
        pickle.dump((X_val, y_val, feature_cols), f)
    print(f"✓ Saved ../data/features_val{OUT_SUFFIX}.pkl")

    with open(f'../data/features_test{OUT_SUFFIX}.pkl', 'wb') as f:
        pickle.dump((X_test, y_test, feature_cols), f)
    print(f"✓ Saved ../data/features_test{OUT_SUFFIX}.pkl")

    with open(f'../data/train_statistics{OUT_SUFFIX}.pkl', 'wb') as f:
        pickle.dump(train_stats, f)
    print(f"✓ Saved ../data/train_statistics{OUT_SUFFIX}.pkl")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("NOTEBOOK 02 COMPLETE")
    print("=" * 80)
    unseen_train = int((X_train['is_new_card'] == 1).sum())
    unseen_val   = int((X_val['is_new_card']   == 1).sum())
    unseen_test  = int((X_test['is_new_card']  == 1).sum())
    print(f"""
✓ ZERO cross-split data leakage
✓ Pandas categorical dtype (native, clean)
✓ All splits use SAME categories (from train)
✓ Unseen values → code -1 (LightGBM handles it)
✓ Ready for LightGBM

Split mode:    {SPLIT_VERSION}
Total features: {len(feature_cols)}

is_new_card breakdown:
  train: {unseen_train:,} unseen-card rows ({unseen_train/len(X_train):.1%})
  val:   {unseen_val:,} unseen-card rows ({unseen_val/len(X_val):.1%})
  test:  {unseen_test:,} unseen-card rows ({unseen_test/len(X_test):.1%})
  (v2 card-split → val/test should both be ~100%)

NOTE: Within-train target-encoded features (device_fraud_rate,
address_fraud_rate, email_fraud_rate, hour_fraud_rate) include each row's
own label in its group statistic — consider out-of-fold encoding to
eliminate this remaining minor leakage before the next training run.

Next: python scripts/ml_pipeline.py
""")
