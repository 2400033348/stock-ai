"""
Experiment 8: new information. Same as Version 2 (Experiment 6, 20-day label) PLUS outside-market features.
 
Run from the project folder (needs internet for the first run; data is cached in data/raw):
    python src/exp8.py
 
New inputs (all from Yahoo Finance):  S&P 500, USD/INR, Brent crude, US VIX.
  * Market-level: 1-day and 5-day change of each, plus the VIX level.
  * Stock-level: how sensitive each stock was to NIFTY, USD/INR and crude over the last 60 days (rolling beta),
    ranked across stocks each day. Example: IT exporters tend to rise when the rupee falls.
TIMING (the important part): India closes before the US does, so a foreign value is treated as known at the close of
day t only if its date is BEFORE t (value lagged by one step). Betas use past data only.
FII/DII flows are not included: there is no free, reliable historical source in Yahoo data.
 
Bar set in advance (same as before, plus an interval): AUC above 0.53 in at least 5 of 8 years, AND the top-10
portfolio beats the equal-weight universe by at least +0.30% per month after costs in at least 5 of 8 years, AND the
95% interval for the net excess return (python src/diagnostics.py results/walk_forward_predictions_v2_h20_exp8.csv)
excludes zero. Compare with Experiment 6 (AUC about 0.51, net excess about -0.14% to -0.24%).
The locked period is never read. Writes results/walk_forward_predictions_v2_h20_exp8.csv
"""
from pathlib import Path
 
import numpy as np
import pandas as pd
 
import v2
from model import PROC_DIR, RES_DIR
 
RAW_DIR = Path("data/raw")
H = 20
EXTERNAL = {"SPX": "^GSPC", "USDINR": "USDINR=X", "BRENT": "BZ=F", "VIX": "^VIX"}
 
 
def fetch_external():
    import yfinance as yf
    out = {}
    for key, ticker in EXTERNAL.items():
        path = RAW_DIR / f"EXT_{key}.parquet"
        if path.exists():
            out[key] = pd.read_parquet(path).iloc[:, 0]
            continue
        df = yf.download(ticker, start="2012-01-01", end="2025-10-15", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        s = df["Close"].dropna().rename(key)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.to_frame().to_parquet(path)
        out[key] = s
        print(f"  downloaded {key} ({ticker}): {len(s)} days")
    return out
 
 
def lag_to(series_or_frame, idx):
    """Value known BEFORE each India date t: forward-fill on the joint calendar, then step back one date."""
    f = series_or_frame
    u = f.index.union(idx)
    return f.reindex(u).ffill().shift(1).reindex(idx)
 
 
def external_features(ext, idx):
    f = pd.DataFrame(index=ext["SPX"].index)
    f["x_spx_1"] = ext["SPX"].pct_change()
    f["x_spx_5"] = ext["SPX"].pct_change(5)
    ux = ext["USDINR"].reindex(ext["SPX"].index.union(ext["USDINR"].index)).ffill()
    f = f.reindex(ux.index)
    f["x_usdinr_1"] = ux.pct_change()
    f["x_usdinr_5"] = ux.pct_change(5)
    bx = ext["BRENT"].reindex(ux.index).ffill()
    f["x_brent_1"] = bx.pct_change()
    f["x_brent_5"] = bx.pct_change(5)
    vx = ext["VIX"].reindex(ux.index).ffill()
    f["x_vix"] = vx
    f["x_vix_5"] = vx.pct_change(5)
    return lag_to(f, idx)
 
 
def stock_betas(raw, nifty_close, ext):
    """Rolling 60-day sensitivity of the stock to NIFTY, USD/INR and Brent, using returns up to day t only."""
    idx = raw.index
    r = raw.Close.pct_change()
    out = pd.DataFrame(index=idx)
    drivers = {"nifty": nifty_close.reindex(idx).ffill().pct_change(),
               "usdinr": ext["USDINR"].reindex(idx).ffill().pct_change(),
               "brent": ext["BRENT"].reindex(idx).ffill().pct_change()}
    for name, d in drivers.items():
        out["beta_" + name] = r.rolling(60).cov(d) / d.rolling(60).var()
    return out
 
 
def main():
    nifty_path = RAW_DIR / "NIFTY.parquet"
    if not nifty_path.exists():
        print("data/raw/NIFTY.parquet not found. Run python src/data_index.py first.")
        return
    nifty_close = pd.read_parquet(nifty_path).Close
    ext = fetch_external()
    names = sorted(p.name.replace("_dataset_rel.parquet", "") for p in PROC_DIR.glob("*_dataset_rel.parquet"))
    dev, betas = {}, {}
    for n in names:
        rp = RAW_DIR / f"{n}.parquet"
        if not rp.exists():
            continue
        raw = pd.read_parquet(rp)
        dev[n] = v2.build_stock(raw, nifty_close, H)
        betas[n] = stock_betas(raw, nifty_close, ext)
    frames, use = v2.prepare(dev, nifty_close)
 
    big = pd.concat([d.assign(stock=n) for n, d in frames.items()])
    bt = pd.concat([betas[n].reindex(frames[n].index).assign(stock=n) for n in frames])
    beta_cols = ["beta_nifty", "beta_usdinr", "beta_brent"]
    for c in beta_cols:
        big[c] = bt[c].values
    rk = big.groupby(level=0)[beta_cols].rank(pct=True)
    for c in beta_cols:
        big["r_" + c] = rk[c].values
    ex = external_features(ext, big.index.unique())
    ex_cols = list(ex.columns)
    for c in ex_cols:
        big[c] = ex.reindex(big.index)[c].values
    use2 = use + ["r_" + c for c in beta_cols] + ex_cols
    big = big[v2.LABEL_COLS + use2 + ["stock"]].dropna()
    frames2 = {n: g.drop(columns="stock") for n, g in big.groupby("stock")}
    print(f"Experiment 8: {len(frames2)} stocks, {len(use2)} features ({len(use2) - len(use)} new), {H}-day label")
    pred = v2.walk_forward_v2(frames2, use2, H)
    if pred.empty:
        return
    RES_DIR.mkdir(exist_ok=True)
    out = RES_DIR / "walk_forward_predictions_v2_h20_exp8.csv"
    pred.to_csv(out)
    print(f"Test period: {pred.index.min().date()} to {pred.index.max().date()} (locked final period NOT used)")
    v2.report(pred, H)
    print(f"\nPredictions saved to {out}")
 
 
if __name__ == "__main__":
    main()