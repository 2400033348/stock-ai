
"""
Week 1, Day 2: download daily prices for the 5 stocks, run basic checks, save to Parquet.
 
Run from the project folder:  python src/data.py
"""
from pathlib import Path
 
import pandas as pd
import yfinance as yf
 
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
START = "2012-01-01"
OUT_DIR = Path("data/raw")
 
 
def download(ticker):
    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]]
 
 
def check(ticker, df):
    print(f"\n=== {ticker} ===")
    print(f"Rows: {len(df)}   From {df.index.min().date()} to {df.index.max().date()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Zero-volume days: {int((df.Volume == 0).sum())}")
    big = df.Close.pct_change().abs()
    jumps = big[big > 0.15]
    print(f"Daily moves above 15%: {len(jumps)}")
    for d, v in jumps.items():
        print(f"   {d.date()}  {100 * v:.1f}%  (check this: real event or bad data?)")
    gaps = df.index.to_series().diff().dt.days
    long_gaps = gaps[gaps > 7]
    print(f"Gaps longer than 7 days: {len(long_gaps)}")
    for d, g in long_gaps.items():
        print(f"   before {d.date()}: {int(g)} days")
 
 
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tk in TICKERS:
        df = download(tk)
        if df.empty:
            print(f"{tk}: no data returned, skipping")
            continue
        df = df.dropna()
        df = df[df.Volume > 0]
        check(tk, df)
        df.to_parquet(OUT_DIR / f"{tk.replace('.NS', '')}.parquet")
    print("\nSaved files:")
    for p in sorted(OUT_DIR.glob("*.parquet")):
        print("  ", p)
 
 
if __name__ == "__main__":
    main()
 

