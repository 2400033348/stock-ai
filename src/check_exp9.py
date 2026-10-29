"""
Experiment 9 follow-up: does the XGBoost ranker add anything beyond simple rules? No retraining.
 
Run from the project folder (after exp9.py):
    python src/check_exp9.py
 
Compares the ranker's top-10 portfolio (same dates, same 0.40% cost, same equal-weight universe) with:
  * the 10 MOST VOLATILE stocks each rebalance (60-day volatility known at the close),
  * the 10 strongest 20-day momentum stocks,
  * 100 random top-10 draws (shows how much luck alone can produce),
and shows how volatile the ranker's picks are compared with the average stock.
Rule set in advance: the ranker adds value only if its average net excess beats BOTH the volatility rule and the
95th percentile of the random draws. Otherwise the +1% is explained by something simpler (or by luck).
"""
import numpy as np
import pandas as pd
from pathlib import Path
 
from v2 import topk_backtest
 
H, COST, K = 20, 0.004, 10
RAW_US = Path("data/raw_us")
PATH = Path("results/walk_forward_predictions_us_h20.csv")
 
 
def excess(pred, col):
    bt = topk_backtest(pred, col, H, K, COST)
    return bt.net - bt.ew, bt
 
 
def main():
    if not PATH.exists():
        print(f"{PATH} not found. Run python src/exp9.py first.")
        return
    pred = pd.read_csv(PATH, index_col=0, parse_dates=True).sort_index()
    pred.index.name = "date"
    vol, mom = {}, {}
    for t in pred.stock.unique():
        c = pd.read_parquet(RAW_US / f"{t}.parquet").Close
        vol[t] = c.pct_change().rolling(60).std()
        mom[t] = c.pct_change(20)
    for name, d in (("vol60", vol), ("mom20", mom)):
        long = pd.DataFrame(d).stack().rename(name).reset_index()
        long.columns = ["date", "stock", name]
        pred = pred.reset_index().merge(long, on=["date", "stock"], how="left").set_index("date").sort_index()
    pred = pred.dropna(subset=["vol60", "mom20"])
    pred["vol_rank"] = pred.groupby(level=0).vol60.rank(pct=True)
 
    rows = {}
    rows["XGBoost ranker"] = excess(pred, "s_rank")
    rows["10 most volatile stocks"] = excess(pred, "vol60")
    rows["10 strongest 20-day momentum"] = excess(pred, "mom20")
    print(f"Dates {pred.index.min().date()} to {pred.index.max().date()}, {pred.stock.nunique()} stocks, "
          f"cost {100 * COST:.2f}% per replaced stock\n")
    print(f"{'Rule':<32}{'Net excess/month':>18}{'Gross':>9}{'Turnover':>10}{'Years +':>9}")
    for name, (ex, bt) in rows.items():
        yrs = ex.groupby(ex.index.year).mean()
        print(f"{name:<32}{100 * ex.mean():>17.3f}%{100 * (bt.gross - bt.ew).mean():>8.3f}%"
              f"{100 * bt.turnover.mean():>9.0f}%{int((yrs > 0).sum()):>6} of {len(yrs)}")
 
    rng = np.random.default_rng(42)
    sims = []
    for _ in range(100):
        p = pred.copy()
        p["rnd"] = rng.random(len(p))
        sims.append(excess(p, "rnd")[0].mean())
    sims = 100 * np.array(sims)
    lo, hi = np.percentile(sims, [5, 95])
    print(f"{'100 random top-10 draws':<32}{sims.mean():>17.3f}%   5th-95th percentile [{lo:+.3f}%, {hi:+.3f}%]")
 
    top = pred.groupby(level=0, group_keys=False).apply(lambda g: g.nlargest(K, "s_rank"))
    print(f"\nRanker's picks: average 60-day-volatility rank {top.vol_rank.mean():.2f} "
          f"(0.50 = an average stock, 1.00 = the most volatile)")
 
    ex_r = rows["XGBoost ranker"][0]
    yrs = ex_r.groupby(ex_r.index.year).mean()
    print("\nRanker net excess by year (% per month): " + " ".join(f"{y}:{100 * v:+.2f}" for y, v in yrs.items()))
    no2020 = ex_r[ex_r.index.year != 2020]
    print(f"Ranker without 2020: {100 * no2020.mean():+.3f}% per month (with 2020: {100 * ex_r.mean():+.3f}%)")
 
    r, v = 100 * ex_r.mean(), 100 * rows["10 most volatile stocks"][0].mean()
    adds = r > v and r > hi
    print(f"\nRule set in advance: ranker {r:+.3f}% vs volatility rule {v:+.3f}% vs random 95th percentile {hi:+.3f}%")
    print("VERDICT: " + ("the ranker beats both simple benchmarks (still not proof: survivorship bias, one market)."
                         if adds else "the ranker does NOT add value beyond a simple rule or luck."))
 
 
if __name__ == "__main__":
    main()
 