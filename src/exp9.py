"""
Experiment 9: the same Version 2 pipeline on US SMALL CAPS (S&P SmallCap 600), benchmark = IWM (Russell 2000 ETF).
 
Run from the project folder (needs internet; first run downloads about 600 stocks and takes a while):
    pip install lxml requests        (once)
    python src/exp9.py
 
Setup:
  * Universe: S&P SmallCap 600 members listed today (Wikipedia). The 100 with the highest average daily dollar volume
    over the last year are kept, among those with price history starting before 2014. SURVIVORSHIP BIAS IS WORSE
    HERE than for NIFTY large caps (today's small caps are the ones that did not fail); results will look flattering.
  * Same features, same 20-day label (beats IWM), same walk-forward 2018-2025, embargo, locked period after 14 Oct 2025.
  * Costs: 0.40% round trip per replaced stock (small caps cost more to trade than Indian large caps).
  * US data is its own market (same time zone for all inputs), so no cross-market lag is needed.
 
Bar set in advance: AUC above 0.53 in at least 5 of 8 years, AND the top-10 portfolio beats the equal-weight universe by
at least +0.30% per month after costs in at least 5 of 8 years, AND the 95% interval of that excess excludes zero.
Writes results/walk_forward_predictions_us_h20.csv
"""
from io import StringIO
from pathlib import Path
 
import numpy as np
import pandas as pd
 
import v2
from diagnostics import block_bootstrap_mean, sharpe
from model import RES_DIR
 
H = 20
COST = 0.004
N_KEEP = 100
RAW_US = Path("data/raw_us")
BENCH = "IWM"
 
 
def sp600_tickers():
    import requests
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
                        headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=30).text
    tables = pd.read_html(StringIO(html))
    for t in tables:
        if "Symbol" in t.columns:
            return sorted({str(s).replace(".", "-") for s in t["Symbol"].dropna()})
    raise RuntimeError("Could not find the ticker table")
 
 
def download_all(tickers):
    import yfinance as yf
    RAW_US.mkdir(parents=True, exist_ok=True)
    out = {}
    todo = []
    for t in tickers + [BENCH]:
        p = RAW_US / f"{t}.parquet"
        if p.exists():
            out[t] = pd.read_parquet(p)
        else:
            todo.append(t)
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        df = yf.download(batch, start="2012-01-01", end="2025-10-15", auto_adjust=True,
                         group_by="ticker", progress=False, threads=True)
        for t in batch:
            try:
                d = df[t][["Open", "High", "Low", "Close", "Volume"]].dropna()
            except Exception:
                continue
            if len(d) > 0:
                d.index = pd.to_datetime(d.index).tz_localize(None)
                d.to_parquet(RAW_US / f"{t}.parquet")
                out[t] = d
        print(f"  downloaded {min(i + 50, len(todo))} of {len(todo)}", flush=True)
    return out
 
 
def pick_universe(data):
    """Keep stocks with history from before 2014, then the N_KEEP with the highest dollar volume in the last year."""
    rows = []
    for t, d in data.items():
        if t == BENCH or d.index.min() > pd.Timestamp("2014-01-01") or len(d) < 2500:
            continue
        last = d[d.index >= d.index.max() - pd.Timedelta(days=365)]
        rows.append((t, (last.Close * last.Volume).mean()))
    rows.sort(key=lambda r: -r[1])
    return [t for t, _ in rows[:N_KEEP]]
 
 
def report_extra(pred):
    print("\n=== Extra: 95% block-bootstrap intervals (cost 0.40%) ===")
    for name, col in v2.SCORES[:3]:
        ic = v2.daily_ic(pred, col)
        bt = v2.topk_backtest(pred, col, H)
        ex = bt.net - bt.ew
        lo, hi = block_bootstrap_mean(ex.values, 3)
        ilo, ihi = block_bootstrap_mean(ic.values, H)
        print(f"{name}: IC {ic.mean():+.4f} [{ilo:+.4f}, {ihi:+.4f}]   net excess {100 * ex.mean():+.3f}% "
              f"[{100 * lo:+.3f}%, {100 * hi:+.3f}%] -> {'excludes zero' if lo > 0 or hi < 0 else 'INCLUDES zero'}   "
              f"information ratio {sharpe(ex, 252 / H):+.2f}")
 
 
def main():
    print("Experiment 9: US small caps")
    tickers = sp600_tickers()
    print(f"  {len(tickers)} tickers in the index list")
    data = download_all(tickers)
    names = pick_universe(data)
    if BENCH not in data or len(names) < 30:
        print("Not enough data downloaded. Check your internet connection and try again.")
        return
    bench = data[BENCH].Close
    dev = {t: v2.build_stock(data[t], bench, H) for t in names}
    frames, use = v2.prepare(dev, bench)
    print(f"{len(frames)} stocks, {len(use)} features, {H}-day label vs {BENCH}")
    orig = v2.topk_backtest
    v2.topk_backtest = lambda pred, col, h, k=None, cost=COST: orig(pred, col, h, k, cost)
    v2.COST = COST
    pred = v2.walk_forward_v2(frames, use, H)
    if pred.empty:
        return
    RES_DIR.mkdir(exist_ok=True)
    out = RES_DIR / "walk_forward_predictions_us_h20.csv"
    pred.to_csv(out)
    print(f"Test period: {pred.index.min().date()} to {pred.index.max().date()} (locked final period NOT used)")
    v2.report(pred, H)
    report_extra(pred)
    print(f"\nPredictions saved to {out}")
 
 
if __name__ == "__main__":
    main()