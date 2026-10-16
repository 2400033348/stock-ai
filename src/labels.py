from pathlib import Path
 
import numpy as np
import pandas as pd
 
STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
HORIZON = 5          # trading days ahead
UP_THRESH = 0.005    # +0.5%
 
 
def make_label(df):
    """Forward 5-day return > 0.5%  ->  1, otherwise 0. NaN where the future is unknown."""
    fwd_ret = df.Close.shift(-HORIZON) / df.Close - 1
    y = (fwd_ret > UP_THRESH).astype(float)
    y[fwd_ret.isna()] = np.nan
    return y, fwd_ret
 
 
def main():
    all_rows = []
    for name in STOCKS:
        raw_path = RAW_DIR / f"{name}.parquet"
        feat_path = OUT_DIR / f"{name}_features.parquet"
        if not raw_path.exists() or not feat_path.exists():
            print(f"{name}: missing file. Run data.py and features.py first.")
            continue
        raw = pd.read_parquet(raw_path)
        feats = pd.read_parquet(feat_path)
 
        y, fwd_ret = make_label(raw)
        data = feats.join(y.rename("y")).join(fwd_ret.rename("fwd_ret"))
        n_before = len(data)
        data = data.dropna()
        data.to_parquet(OUT_DIR / f"{name}_dataset.parquet")
 
        share_up = data.y.mean()
        print(f"\n=== {name} ===")
        print(f"Rows with features: {n_before}   Rows with label: {len(data)}   "
              f"(last {n_before - len(data)} days have no answer yet)")
        print(f"Label = 1 (rose > 0.5% in 5 days): {int(data.y.sum())} days "
              f"= {100 * share_up:.1f}%")
        print(f"Label = 0: {int((1 - data.y).sum())} days = {100 * (1 - share_up):.1f}%")
        all_rows.append(data.assign(stock=name))
 
    if not all_rows:
        return
    pooled = pd.concat(all_rows)
    print("\n=== Share of label = 1 by year (all 5 stocks together) ===")
    by_year = pooled.groupby(pooled.index.year).y.agg(["mean", "count"])
    for year, row in by_year.iterrows():
        print(f"{year}: {100 * row['mean']:5.1f}%   ({int(row['count'])} rows)")
    print(f"\nOverall: {100 * pooled.y.mean():.1f}% of days are label = 1")
    print("A model that always says 'up' would be right this often (the baseline to beat).")
 
 
if __name__ == "__main__":
    main()