"""
Experiment 2, step 1: download the NIFTY 50 index (benchmark for the relative label).

Run from the project folder:  python src/data_index.py
Writes data/raw/NIFTY.parquet

The 5 stock files are NOT re-downloaded, so Experiment 1 stays exactly comparable
(re-downloading adjusted prices can change old values slightly).
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

START = "2012-01-01"
TICKER = "^NSEI"
RAW_DIR = Path("data/raw")
STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    print(f"NIFTY 50: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
    print(f"Missing Close values: {int(df.Close.isna().sum())}")
    gaps = df.index.to_series().diff().dt.days
    print(f"Gaps longer than 7 days: {int((gaps > 7).sum())}")

    for st in STOCKS:
        p = RAW_DIR / f"{st}.parquet"
        if p.exists():
            stock_dates = pd.read_parquet(p).index
            missing = int((~stock_dates.isin(df.index)).sum())
            print(f"{st}: {missing} stock trading days not found in NIFTY data")

    df.to_parquet(RAW_DIR / "NIFTY.parquet")
    print(f"Saved {RAW_DIR / 'NIFTY.parquet'}")


if __name__ == "__main__":
    main()