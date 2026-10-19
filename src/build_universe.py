"""
Experiment 3b, step 1: add more stocks (raw prices, features, relative label).

Run from the project folder (after data_index.py):  python src/build_universe.py
Writes data/raw/<STOCK>.parquet and data/processed/<STOCK>_dataset_rel.parquet for NEW stocks only.
Stocks that already have a dataset file (your original 5) are skipped, never re-downloaded.

IMPORTANT, read before trusting any result:
  * This is a list of large caps that exist TODAY. Stocks that were dropped from the index over the
    years (often the poor performers) are missing. That is survivorship bias: results on this list
    will look better than they would have in real time. Treat any edge found here with suspicion.
  * Check the list against nseindia.com. Index membership changes. Edit UNIVERSE below freely.
  * Tickers that fail to download, or have too little history, are skipped and printed.
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from features import make_features
from labels_rel import make_label_rel

START = "2012-01-01"
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
MIN_ROWS = 1000          # feature rows with labels needed to include a stock

UNIVERSE = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "INDUSINDBK", "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "SBILIFE", "SBIN",
    "SUNPHARMA", "TATACONSUM", "TATASTEEL", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO",
]


def file_key(symbol):
    return symbol.replace("&", "_")


def build_dataset(raw, nifty_close):
    """Features + relative label for one stock. Same logic as Experiment 2."""
    feats = make_features(raw).dropna()
    y, s_fwd, n_fwd, excess = make_label_rel(raw, nifty_close)
    data = (feats.join(y.rename("y")).join(s_fwd.rename("fwd_ret"))
                 .join(n_fwd.rename("nifty_fwd_ret")).join(excess.rename("excess_ret")))
    return data.dropna()


def main():
    nifty_path = RAW_DIR / "NIFTY.parquet"
    if not nifty_path.exists():
        print("data/raw/NIFTY.parquet not found. Run python src/data_index.py first.")
        return
    nifty_close = pd.read_parquet(nifty_path).Close
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    added, skipped = [], []
    for sym in UNIVERSE:
        key = file_key(sym)
        if (PROC_DIR / f"{key}_dataset_rel.parquet").exists():
            print(f"{sym}: already built, skipped")
            continue
        try:
            raw = yf.download(f"{sym}.NS", start=START, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            raw = raw[raw.Volume > 0]
        except Exception as e:
            skipped.append((sym, f"download failed: {e}"))
            continue
        if len(raw) < MIN_ROWS + 60:
            skipped.append((sym, f"only {len(raw)} rows"))
            continue
        data = build_dataset(raw, nifty_close)
        if len(data) < MIN_ROWS:
            skipped.append((sym, f"only {len(data)} usable rows"))
            continue
        big = raw.Close.pct_change().abs()
        n_big = int((big > 0.25).sum())
        raw.to_parquet(RAW_DIR / f"{key}.parquet")
        data.to_parquet(PROC_DIR / f"{key}_dataset_rel.parquet")
        added.append(sym)
        flag = f"   WARNING: {n_big} daily moves above 25%, check for bad data" if n_big else ""
        print(f"{sym}: {len(data)} rows from {data.index.min().date()}{flag}")
        time.sleep(0.5)

    print(f"\nAdded {len(added)} stocks. Skipped {len(skipped)}:")
    for sym, why in skipped:
        print(f"   {sym}: {why}")
    total = len(list(PROC_DIR.glob('*_dataset_rel.parquet')))
    print(f"Total stocks with a dataset file now: {total}")


if __name__ == "__main__":
    main()