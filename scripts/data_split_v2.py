"""
V2 Data Split — Card-based (group) split.

Every transaction for a given card goes to exactly one split, so the model
can never memorize card-specific patterns from training and apply them to the
same card at evaluation time.

Split is stratified by fraud-card vs clean-card to preserve the ~3.5% fraud
rate across all three sets.

Run from project root:
    python scripts/data_split_v2.py

Outputs:
    data/ieee_prepared_v2.pkl   — {train, val, test} DataFrames (same schema
                                   as ieee_prepared.pkl, drop-in for FE script)

Then run feature engineering against v2 data:
    python scripts/feature_engineering.py --split v2
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

RATIOS = (0.70, 0.10, 0.20)   # train / val / test by unique cards
SEED   = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raw(data_dir: Path) -> pd.DataFrame:
    """Merge transaction + identity CSVs (same join as temporal v1)."""
    tx  = pd.read_csv(data_dir / "train_transaction.csv")
    idf = pd.read_csv(data_dir / "train_identity.csv")
    df  = tx.merge(idf, on="TransactionID", how="left")
    # Keep chronological order within each card's history for D-column features
    df  = df.sort_values("TransactionDT").reset_index(drop=True)
    return df


def stratified_card_split(
    df: pd.DataFrame,
    ratios: tuple = RATIOS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split by card1, stratified over fraud-card vs clean-card groups.

    Why stratify:
        Fraud transactions are concentrated on a subset of cards. A naive
        random shuffle of cards can put most fraud cards in one split, making
        fraud rates inconsistent across train / val / test.
        Stratifying ensures each split gets ~70/10/20 of the *fraud cards*
        and ~70/10/20 of the *clean cards* independently.

    Rows with null card1 are assigned to train (no group to place them in).
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "Ratios must sum to 1"

    rng = np.random.default_rng(seed)

    fraud_cards = set(df.loc[df["isFraud"] == 1, "card1"].dropna().unique())
    all_cards   = set(df["card1"].dropna().unique())
    clean_cards = all_cards - fraud_cards

    def _split(cards: set) -> tuple[set, set, set]:
        arr = rng.permutation(sorted(cards))
        n   = len(arr)
        i1  = int(ratios[0] * n)
        i2  = int((ratios[0] + ratios[1]) * n)
        return set(arr[:i1]), set(arr[i1:i2]), set(arr[i2:])

    f_train, f_val, f_test = _split(fraud_cards)
    c_train, c_val, c_test = _split(clean_cards)

    train_cards = f_train | c_train
    val_cards   = f_val   | c_val
    test_cards  = f_test  | c_test

    null_mask = df["card1"].isna()
    train_df = df[df["card1"].isin(train_cards) | null_mask].copy()
    val_df   = df[df["card1"].isin(val_cards)].copy()
    test_df  = df[df["card1"].isin(test_cards)].copy()

    meta = {
        "train_cards":       len(train_cards),
        "val_cards":         len(val_cards),
        "test_cards":        len(test_cards),
        "fraud_cards_total": len(fraud_cards),
        "fraud_cards_train": len(f_train),
        "fraud_cards_val":   len(f_val),
        "fraud_cards_test":  len(f_test),
    }
    return train_df, val_df, test_df, meta


def verify_no_card_overlap(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    t = set(train["card1"].dropna().unique())
    v = set(val["card1"].dropna().unique())
    e = set(test["card1"].dropna().unique())
    assert len(t & v) == 0, f"Card leak train∩val: {len(t & v)}"
    assert len(t & e) == 0, f"Card leak train∩test: {len(t & e)}"
    assert len(v & e) == 0, f"Card leak val∩test: {len(v & e)}"


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root     = Path(__file__).parent.parent
    data_dir = root / "data"

    print("=" * 80)
    print("DATA SPLIT V2: STRATIFIED CARD-BASED SPLIT")
    print("=" * 80)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1] Loading raw IEEE-CIS data...")
    df = load_raw(data_dir)
    print(f"  Loaded {len(df):,} transactions, {df.shape[1]} columns")
    print(f"  Date range (TransactionDT): {df['TransactionDT'].min():,} – {df['TransactionDT'].max():,}")
    print(f"  Unique cards (card1):       {df['card1'].nunique():,}")
    print(f"  Overall fraud rate:         {df['isFraud'].mean():.3%}")
    print(f"  Fraud cards:                {df.loc[df['isFraud']==1,'card1'].nunique():,} "
          f"/ {df['card1'].nunique():,} total")

    # ── Split ─────────────────────────────────────────────────────────────────
    print(f"\n[2] Stratified card split {RATIOS[0]:.0%} / {RATIOS[1]:.0%} / {RATIOS[2]:.0%}...")
    train, val, test, meta = stratified_card_split(df)

    header = f"  {'Split':<8} {'Rows':>10} {'Cards':>8} {'Fraud cards':>12} {'Fraud rate':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, split_df, n_cards, n_fraud_cards in [
        ("train", train, meta["train_cards"], meta["fraud_cards_train"]),
        ("val",   val,   meta["val_cards"],   meta["fraud_cards_val"]),
        ("test",  test,  meta["test_cards"],  meta["fraud_cards_test"]),
    ]:
        print(f"  {name:<8} {len(split_df):>10,} {n_cards:>8,} "
              f"{n_fraud_cards:>12,} {split_df['isFraud'].mean():>11.3%}")

    # ── Verify ────────────────────────────────────────────────────────────────
    print("\n[3] Verifying zero card overlap across splits...")
    verify_no_card_overlap(train, val, test)
    print("  ✓ No card appears in more than one split")

    # ── Key difference vs temporal split ─────────────────────────────────────
    train_card_set = set(train["card1"].dropna().unique())
    val_unseen   = (~val["card1"].isin(train_card_set)).mean()
    test_unseen  = (~test["card1"].isin(train_card_set)).mean()
    print(f"\n[4] Unseen-card rate vs train:")
    print(f"  Val  transactions on cards not in train: {val_unseen:.1%}  (should be ~100%)")
    print(f"  Test transactions on cards not in train: {test_unseen:.1%}  (should be ~100%)")
    print(f"\n  Compare to temporal split where many val/test cards ARE seen in train.")
    print(f"  Card-based split is a STRICTER evaluation of generalisation.")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[5] Saving...")
    out_path = data_dir / "ieee_prepared_v2.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"train": train, "val": val, "test": test}, f)
    print(f"  ✓ Saved {out_path}")
    print(f"\n  Next step:")
    print(f"    python scripts/feature_engineering.py --split v2")

    print("\n" + "=" * 80)
    print("SPLIT V2 COMPLETE")
    print("=" * 80)
