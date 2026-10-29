
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from model import (FIRST_TEST_YEAR, HORIZON, MIN_TRAIN_ROWS, NON_FEATURES, PROC_DIR,
                   RES_DIR, calibration_table, dev_data, make_models, safe_auc)
 
CORE5 = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
 
 
def load_dev(names):
    out = {}
    for n in names:
        p = PROC_DIR / f"{n}_dataset_rel.parquet"
        if p.exists():
            out[n] = dev_data(pd.read_parquet(p))
    return out
 
 
def walk_forward_pooled(dev_by_stock, first_year=FIRST_TEST_YEAR, feature_cols=None):
    last_year = max(d.index.year.max() for d in dev_by_stock.values())
    parts = []
    for year in range(first_year, last_year + 1):
        jan1 = f"{year}-01-01"
        train = pd.concat([d[d.index < jan1].iloc[:-HORIZON] for d in dev_by_stock.values()])
        test = pd.concat([d[d.index.year == year].assign(stock=n) for n, d in dev_by_stock.items()])
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        cols = feature_cols or [c for c in train.columns if c not in NON_FEATURES]
        out = pd.DataFrame({"y": test.y, "fwd_ret": test.fwd_ret, "excess_ret": test.excess_ret,
                            "mom5": test.ret_5, "stock": test.stock}, index=test.index)
        for name, model in make_models().items():
            model.fit(train[cols], train.y)
            out[f"prob_{name}"] = model.predict_proba(test[cols])[:, 1]
        parts.append(out)
        print(f"  {year}: trained on {len(train)} rows from {len(dev_by_stock)} stocks, "
              f"tested on {len(test)}", flush=True)
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()
 
 
def spread_table(pred, col):
    """Each day: rank the stocks by prediction; compare the top fifth with the bottom fifth."""
    rows = []
    for d, g in pred.groupby(level=0):
        n = len(g)
        if n < 5:
            continue
        k = max(1, n // 5)
        g = g.sort_values(col)
        rows.append((d, g.excess_ret.iloc[-k:].mean(), g.excess_ret.iloc[:k].mean(), g.excess_ret.mean()))
    t = pd.DataFrame(rows, columns=["date", "top", "bottom", "all"]).set_index("date")
    t["top_vs_all"] = t.top - t["all"]
    t["spread"] = t.top - t.bottom
    return t
 
 
def report(pred):
    models = [("Random Forest", "prob_rf"), ("XGBoost", "prob_xgb"), ("Momentum (ret_5)", "mom5")]
    n_stocks = pred.stock.nunique()
    print(f"\nStocks in the test: {n_stocks}   rows: {len(pred)}   "
          f"share label = 1: {100 * pred.y.mean():.1f}%")
 
    print("\n=== AUC by test year (0.50 = no skill) ===")
    print(f"{'Year':<6}{'Rows':>8}" + "".join(f"{n:>20}" for n, _ in models))
    for year, g in pred.groupby(pred.index.year):
        print(f"{year:<6}{len(g):>8}" + "".join(f"{safe_auc(g.y, g[c]):>20.3f}" for _, c in models))
    print(f"{'ALL':<6}{len(pred):>8}" + "".join(f"{safe_auc(pred.y, pred[c]):>20.3f}" for _, c in models))
    for _, c in models[:2]:
        per_year = pred.groupby(pred.index.year).apply(lambda g: safe_auc(g.y, g[c]))
        print(f"   {c}: years with AUC above 0.53: {int((per_year > 0.53).sum())} of {len(per_year)}")
 
    if n_stocks > 6:
        st_auc = pred.groupby("stock").apply(lambda g: safe_auc(g.y, g.prob_rf)).dropna()
        print(f"\nRandom Forest AUC per stock: mean {st_auc.mean():.3f}, min {st_auc.min():.3f}, "
              f"max {st_auc.max():.3f}, stocks above 0.50: {int((st_auc > 0.5).sum())} of {len(st_auc)}")
 
    for name, col in models[:2]:
        print(f"\n=== Calibration: {name} ===")
        print(calibration_table(pred.y, pred[col]).round(3).to_string())
        hi = pred[pred[col] >= 0.55]
        if len(hi):
            print(f"Signals with probability >= 0.55: {len(hi)} ({100 * len(hi) / len(pred):.1f}% of rows)   "
                  f"hit rate {100 * hi.y.mean():.1f}%")
            print(f"   Average 5-day return vs NIFTY on these signals: {100 * hi.excess_ret.mean():+.3f}%  "
                  f"(all rows: {100 * pred.excess_ret.mean():+.3f}%)  "
                  f"-> advantage {100 * (hi.excess_ret.mean() - pred.excess_ret.mean()):+.3f}% per 5 days, before costs")
 
        t = spread_table(pred, col)
        if not t.empty:
            print(f"Top fifth vs bottom fifth of stocks each day ({name}), average 5-day return vs NIFTY:")
            print(f"   top {100 * t.top.mean():+.3f}%   bottom {100 * t.bottom.mean():+.3f}%   "
                  f"all {100 * t['all'].mean():+.3f}%   top-minus-bottom {100 * t.spread.mean():+.3f}%   "
                  f"top-minus-all {100 * t.top_vs_all.mean():+.3f}%")
            by_year = (100 * t.groupby(t.index.year)[["top_vs_all", "spread"]].mean()).round(3)
            print("   by year (top-minus-all / top-minus-bottom, % per 5 days):")
            print("   " + "   ".join(f"{y}: {r.top_vs_all:+.2f}/{r.spread:+.2f}" for y, r in by_year.iterrows()))
    print("\nBar set in advance: AUC > 0.53 in at least 5 of 8 years, and an advantage of at least "
          "+0.30% per 5 days before costs.")
 
 
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "5"
    if which == "5":
        names = CORE5
    else:
        names = sorted(p.name.replace("_dataset_rel.parquet", "")
                       for p in PROC_DIR.glob("*_dataset_rel.parquet"))
    dev = load_dev(names)
    if not dev:
        print("No *_dataset_rel.parquet files found. Run labels_rel.py (and build_universe.py).")
        return
    print(f"Experiment 3{'a' if which == '5' else 'b'}: one pooled model, {len(dev)} stocks")
    pred = walk_forward_pooled(dev)
    if pred.empty:
        return
    RES_DIR.mkdir(exist_ok=True)
    out = RES_DIR / f"walk_forward_predictions_pooled_{'5' if which == '5' else 'all'}.csv"
    pred.to_csv(out)
    print(f"Test period: {pred.index.min().date()} to {pred.index.max().date()} "
          "(locked final period NOT used)")
    report(pred)
    print(f"\nPredictions saved to {out}")
 
 
if __name__ == "__main__":
    main()