"""
Experiment 2, step 2: build the RELATIVE label and join it with the existing features.

Run from the project folder (after data_index.py):  python src/labels_rel.py
Reads  data/raw/<STOCK>.parquet, data/raw/NIFTY.parquet, data/processed/<STOCK>_features.parquet
Writes data/processed/<STOCK>_dataset_rel.parquet

Relative label y on day t:
    1 if the stock's 5-day return is HIGHER than NIFTY 50's 5-day return, else 0
(Features are unchanged from Experiment 1. Only the label is different.)
"""
from pathlib import Path

import numpy as np
import pandas as pd

STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
HORIZON = 5


def make_label_rel(stock, nifty_close):
    """Return (y, stock_fwd_ret, nifty_fwd_ret, excess_ret). NaN where the future is unknown."""
    nc = nifty_close.reindex(stock.index).ffill()
    stock_fwd = stock.Close.shift(-HORIZON) / stock.Close - 1
    nifty_fwd = nc.shift(-HORIZON) / nc - 1
    excess = stock_fwd - nifty_fwd
    y = (excess > 0).astype(float)
    y[excess.isna()] = np.nan
    return y, stock_fwd, nifty_fwd, excess


def main():
    nifty_path = RAW_DIR / "NIFTY.parquet"
    if not nifty_path.exists():
        print(f"{nifty_path} not found. Run python src/data_index.py first.")
        return
    nifty = pd.read_parquet(nifty_path)

    pooled = []
    for name in STOCKS:
        raw_path = RAW_DIR / f"{name}.parquet"
        feat_path = OUT_DIR / f"{name}_features.parquet"
        if not raw_path.exists() or not feat_path.exists():
            print(f"{name}: missing file. Run data.py and features.py first.")
            continue
        raw = pd.read_parquet(raw_path)
        feats = pd.read_parquet(feat_path)

        y, s_fwd, n_fwd, excess = make_label_rel(raw, nifty.Close)
        data = (feats.join(y.rename("y")).join(s_fwd.rename("fwd_ret"))
                     .join(n_fwd.rename("nifty_fwd_ret")).join(excess.rename("excess_ret")))
        n_before = len(data)
        data = data.dropna()
        data.to_parquet(OUT_DIR / f"{name}_dataset_rel.parquet")

        print(f"\n=== {name} ===")
        print(f"Rows with features: {n_before}   Rows with label: {len(data)}")
        print(f"Label = 1 (beat NIFTY over 5 days): {100 * data.y.mean():.1f}%   "
              f"average excess return: {100 * data.excess_ret.mean():.3f}%")
        pooled.append(data)

    if not pooled:
        return
    allrows = pd.concat(pooled)
    print("\n=== Share of label = 1 by year (all 5 stocks together) ===")
    for year, g in allrows.groupby(allrows.index.year):
        print(f"{year}: {100 * g.y.mean():5.1f}%   ({len(g)} rows)")
    print(f"\nOverall: {100 * allrows.y.mean():.1f}% of days are label = 1 (near 50% is what we want)")


if __name__ == "__main__":
    main()