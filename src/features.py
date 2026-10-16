from pathlib import Path
 
import pandas as pd
 
STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
ATR_N = 14
 
 
def rsi(close, n=14):
    """Relative Strength Index: 0-100. High = rose fast recently, low = fell fast."""
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down)
 
 
def atr(df, n=ATR_N):
    """Average True Range: the stock's typical daily price range over n days."""
    prev_close = df.Close.shift(1)
    true_range = pd.concat([
        df.High - df.Low,
        (df.High - prev_close).abs(),
        (df.Low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(n).mean()
 
 
def make_features(df):
    """Every feature on day t uses only data up to and including day t."""
    c = df.Close
    f = pd.DataFrame(index=df.index)
 
    # Returns: how much the price changed over the last n days
    for n in (1, 3, 5, 10, 20):
        f[f"ret_{n}"] = c.pct_change(n)
 
    # Trend: how far the price is above/below its moving average
    for n in (10, 20, 50):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
 
    # Momentum
    f["rsi14"] = rsi(c)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    f["macd"] = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
 
    # Volatility: how jumpy the price is
    f["vol20"] = c.pct_change().rolling(20).std()
    f["atr_pct"] = atr(df) / c
    f["range_pct"] = (df.High - df.Low) / c
 
    # Volume compared with its 20-day normal
    f["vol_ratio"] = df.Volume / df.Volume.rolling(20).mean()
 
    # Gap between today's open and yesterday's close
    f["gap"] = df.Open / c.shift(1) - 1
    return f
 
 
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in STOCKS:
        path = RAW_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"{name}: {path} not found. Run python src/data.py first.")
            continue
        df = pd.read_parquet(path)
        f = make_features(df)
        warmup_rows = int(f.isna().any(axis=1).sum())
        f = f.dropna()
        f.to_parquet(OUT_DIR / f"{name}_features.parquet")
        print(f"\n=== {name} ===")
        print(f"Raw rows: {len(df)}   Feature rows: {len(f)}   "
              f"Dropped (warm-up): {warmup_rows}")
        print(f"Features ({f.shape[1]}): {', '.join(f.columns)}")
        print(f.tail(3).round(4).T)
 
 
if __name__ == "__main__":
    main()