import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 
import pooled as P
from features import make_features
from model import DEV_END, PROC_DIR, RES_DIR, calibration_table, safe_auc
 
RAW_DIR = Path("data/raw")
COST = 0.003
 
 
def make_label_h(stock, nifty_close, h):
    """1 if the stock's h-day return beats NIFTY's h-day return. NaN where the future is unknown."""
    nc = nifty_close.reindex(stock.index).ffill()
    s_fwd = stock.Close.shift(-h) / stock.Close - 1
    n_fwd = nc.shift(-h) / nc - 1
    excess = s_fwd - n_fwd
    y = (excess > 0).astype(float)
    y[excess.isna()] = np.nan
    return y, s_fwd, n_fwd, excess
 
 
def build_dev_from_raw(raw, nifty_close, h):
    """Features + h-day relative label, development rows only (labels never reach past DEV_END)."""
    feats = make_features(raw).dropna()
    y, s_fwd, n_fwd, excess = make_label_h(raw, nifty_close, h)
    data = (feats.join(y.rename("y")).join(s_fwd.rename("fwd_ret"))
                 .join(n_fwd.rename("nifty_fwd_ret")).join(excess.rename("excess_ret"))).dropna()
    return data[data.index <= DEV_END].iloc[:-h]
 
 
def report(pred, h):
    models = [("Random Forest", "prob_rf"), ("XGBoost", "prob_xgb"), ("Momentum (ret_5)", "mom5")]
    print(f"\nStocks in the test: {pred.stock.nunique()}   rows: {len(pred)}   "
          f"share label = 1: {100 * pred.y.mean():.1f}%")
 
    print(f"\n=== AUC by test year, {h}-day label (0.50 = no skill) ===")
    print(f"{'Year':<6}{'Rows':>8}" + "".join(f"{n:>20}" for n, _ in models))
    for year, g in pred.groupby(pred.index.year):
        print(f"{year:<6}{len(g):>8}" + "".join(f"{safe_auc(g.y, g[c]):>20.3f}" for _, c in models))
    print(f"{'ALL':<6}{len(pred):>8}" + "".join(f"{safe_auc(pred.y, pred[c]):>20.3f}" for _, c in models))
    for _, c in models[:2]:
        per_year = pred.groupby(pred.index.year).apply(lambda g: safe_auc(g.y, g[c]))
        print(f"   {c}: years with AUC above 0.53: {int((per_year > 0.53).sum())} of {len(per_year)}")
 
    days = np.sort(pred.index.unique())
    sub = pred[pred.index.isin(days[::h])]
    print(f"\nAUC on non-overlapping dates only (every {h}th day, {len(sub)} rows): "
          f"RF {safe_auc(sub.y, sub.prob_rf):.3f}   XGB {safe_auc(sub.y, sub.prob_xgb):.3f}")
    print("   (labels overlap heavily at long horizons, so this is the more honest, noisier number)")
 
    st_auc = pred.groupby("stock").apply(lambda g: safe_auc(g.y, g.prob_rf)).dropna()
    print(f"Random Forest AUC per stock: mean {st_auc.mean():.3f}, min {st_auc.min():.3f}, "
          f"max {st_auc.max():.3f}, stocks above 0.50: {int((st_auc > 0.5).sum())} of {len(st_auc)}")
 
    for name, col in models[:2]:
        print(f"\n=== Calibration: {name} ===")
        print(calibration_table(pred.y, pred[col]).round(3).to_string())
        hi = pred[pred[col] >= 0.55]
        if len(hi):
            print(f"Signals with probability >= 0.55: {len(hi)} ({100 * len(hi) / len(pred):.1f}% of rows)   "
                  f"hit rate {100 * hi.y.mean():.1f}%")
            print(f"   Average {h}-day return vs NIFTY on these signals: {100 * hi.excess_ret.mean():+.3f}%  "
                  f"(all rows: {100 * pred.excess_ret.mean():+.3f}%)  "
                  f"-> advantage {100 * (hi.excess_ret.mean() - pred.excess_ret.mean()):+.3f}% per {h} days, before costs")
        t = P.spread_table(pred, col)
        if not t.empty:
            print(f"Top fifth vs bottom fifth of stocks each day ({name}), average {h}-day return vs NIFTY:")
            print(f"   top {100 * t.top.mean():+.3f}%   bottom {100 * t.bottom.mean():+.3f}%   "
                  f"all {100 * t['all'].mean():+.3f}%   top-minus-bottom {100 * t.spread.mean():+.3f}%   "
                  f"top-minus-all {100 * t.top_vs_all.mean():+.3f}%")
            by_year = (100 * t.groupby(t.index.year)[["top_vs_all", "spread"]].mean()).round(3)
            print(f"   by year (top-minus-all / top-minus-bottom, % per {h} days):")
            print("   " + "   ".join(f"{y}: {r.top_vs_all:+.2f}/{r.spread:+.2f}" for y, r in by_year.iterrows()))
 
    print(f"\nBar set in advance: AUC > 0.53 in at least 5 of 8 years, and an advantage of at least "
          f"+{100 * COST:.2f}% per {h} days before costs (one round trip per holding period).")
 
 
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("10", "20"):
        print("Use: python src/horizon.py 10   or   python src/horizon.py 20")
        return
    h = int(sys.argv[1])
    nifty_path = RAW_DIR / "NIFTY.parquet"
    if not nifty_path.exists():
        print("data/raw/NIFTY.parquet not found. Run python src/data_index.py first.")
        return
    nifty_close = pd.read_parquet(nifty_path).Close
    names = sorted(p.name.replace("_dataset_rel.parquet", "")
                   for p in PROC_DIR.glob("*_dataset_rel.parquet"))
    dev = {}
    for n in names:
        raw_path = RAW_DIR / f"{n}.parquet"
        if raw_path.exists():
            dev[n] = build_dev_from_raw(pd.read_parquet(raw_path), nifty_close, h)
        else:
            print(f"{n}: {raw_path} not found, skipped")
    if not dev:
        print("No stock data found. Run build_universe.py first.")
        return
    print(f"Experiment 5: {h}-day label, one pooled model, {len(dev)} stocks")
    P.HORIZON = h            # makes the embargo in pooled.walk_forward_pooled h rows long
    pred = P.walk_forward_pooled(dev)
    if pred.empty:
        return
    RES_DIR.mkdir(exist_ok=True)
    out = RES_DIR / f"walk_forward_predictions_h{h}.csv"
    pred.to_csv(out)
    print(f"Test period: {pred.index.min().date()} to {pred.index.max().date()} "
          "(locked final period NOT used)")
    report(pred, h)
    print(f"\nPredictions saved to {out}")
 
 
if __name__ == "__main__":
    main()
 