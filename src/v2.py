import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRanker
 
from features import make_features
from horizon import make_label_h
from model import DEV_END, FIRST_TEST_YEAR, MIN_TRAIN_ROWS, PROC_DIR, RES_DIR, SEED, safe_auc
 
RAW_DIR = Path("data/raw")
K = 10                       # stocks held in the portfolio
COST = 0.003                 # round trip, paid on every stock that is replaced
LABEL_COLS = ["y", "fwd_ret", "nifty_fwd_ret", "excess_ret"]
SCORES = [("Logistic Regression", "s_logit"), ("XGBoost classifier", "s_xgb"),
          ("XGBoost ranker", "s_rank"), ("Momentum rank", "mom")]
 
 
# ---------------------------------------------------------------- features
def long_features(raw):
    """Longer-term features. Each uses only data up to and including day t."""
    c = raw.Close
    f = pd.DataFrame(index=raw.index)
    f["ret_60"] = c.pct_change(60)
    f["ret_120"] = c.pct_change(120)
    f["mom_12_1"] = c.shift(20) / c.shift(250) - 1          # 12-month return, skipping the last month
    f["dist_52w_high"] = c / c.rolling(250).max() - 1
    f["dist_52w_low"] = c / c.rolling(250).min() - 1
    f["vol60"] = c.pct_change().rolling(60).std()
    f["vol_chg"] = raw.Volume.rolling(5).mean() / raw.Volume.rolling(60).mean()
    return f
 
 
def market_feats(close):
    """NIFTY regime features known at the close of day t."""
    r = close.pct_change()
    f = pd.DataFrame(index=close.index)
    f["m_ret_5"] = close.pct_change(5)
    f["m_ret_20"] = close.pct_change(20)
    f["m_ret_60"] = close.pct_change(60)
    f["m_dist_sma50"] = close / close.rolling(50).mean() - 1
    f["m_dist_sma200"] = close / close.rolling(200).mean() - 1
    f["m_vol20"] = r.rolling(20).std()
    return f
 
 
def build_stock(raw, nifty_close, h):
    """One stock: features + h-day relative label. Development rows only; labels never pass DEV_END."""
    feats = pd.concat([make_features(raw), long_features(raw)], axis=1)
    y, s_fwd, n_fwd, excess = make_label_h(raw, nifty_close, h)
    data = (feats.join(y.rename("y")).join(s_fwd.rename("fwd_ret"))
                 .join(n_fwd.rename("nifty_fwd_ret")).join(excess.rename("excess_ret"))).dropna()
    return data[data.index <= DEV_END].iloc[:-h]
 
 
def prepare(dev, nifty_close):
    """Ranks across stocks (same day only), breadth and NIFTY regime. Returns per-stock frames."""
    big = pd.concat([d.assign(stock=n) for n, d in dev.items()])
    stock_cols = [c for c in big.columns if c not in LABEL_COLS + ["stock"]]
    big["breadth50"] = (big.dist_sma50 > 0).groupby(level=0).transform("mean")
    big["breadth_ret20"] = (big.ret_20 > 0).groupby(level=0).transform("mean")
    rk = big.groupby(level=0)[stock_cols].rank(pct=True)
    for c in stock_cols:
        big["r_" + c] = rk[c].values
    mk = market_feats(nifty_close)
    mk = mk.reindex(mk.index.union(big.index.unique())).ffill().reindex(big.index)
    for c in mk.columns:
        big[c] = mk[c].values
    use = ["r_" + c for c in stock_cols] + list(mk.columns) + ["breadth50", "breadth_ret20"]
    big = big[LABEL_COLS + use + ["stock"]].dropna()
    frames = {n: g.drop(columns="stock") for n, g in big.groupby("stock")}
    return frames, use
 
 
# ---------------------------------------------------------------- models
def make_classifiers():
    return {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=300)),
        "xgb": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=20, random_state=SEED,
                             n_jobs=-1, eval_metric="logloss"),
    }
 
 
def make_ranker():
    return XGBRanker(objective="rank:pairwise", n_estimators=200, max_depth=3, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                     random_state=SEED, n_jobs=-1)
 
 
def walk_forward_v2(frames, use, h, first_year=FIRST_TEST_YEAR):
    mom_col = "r_ret_5" if h == 5 else "r_ret_20"
    last_year = max(d.index.year.max() for d in frames.values())
    parts = []
    for year in range(first_year, last_year + 1):
        jan1 = f"{year}-01-01"
        train = pd.concat([d[d.index < jan1].iloc[:-h] for d in frames.values()])      # embargo: h rows
        test = pd.concat([d[d.index.year == year].assign(stock=n) for n, d in frames.items()])
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        out = test[LABEL_COLS + ["stock"]].copy()
        out["mom"] = test[mom_col]
        for name, model in make_classifiers().items():
            model.fit(train[use], train.y)
            out["s_" + name] = model.predict_proba(test[use])[:, 1]
        tr = train.sort_index(kind="stable")                    # ranker needs rows grouped by date
        qid = pd.factorize(tr.index)[0]
        rel = tr.groupby(level=0).excess_ret.rank(pct=True).values
        rel = np.minimum((rel * 5).astype(int), 4)              # 0 (worst fifth) ... 4 (best fifth)
        ranker = make_ranker()
        ranker.fit(tr[use], rel, qid=qid)
        out["s_rank"] = ranker.predict(test[use])
        parts.append(out)
        print(f"  {year}: trained on {len(train)} rows, tested on {len(test)}", flush=True)
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()
 
 
# ---------------------------------------------------------------- evaluation
def daily_ic(pred, col):
    """Rank correlation between the score and the realised excess return, one value per day."""
    def one(g):
        if len(g) < 10:
            return np.nan
        return np.corrcoef(g[col].rank(), g.excess_ret.rank())[0, 1]
    return pred.groupby(level=0).apply(one).dropna()
 
 
def topk_backtest(pred, col, h, k=None, cost=COST):
    """Rebalance every h trading days (no overlapping periods), hold the k best-scored stocks, equal weight."""
    k = k or K
    dates = np.sort(pred.index.unique())[::h]
    prev, rows = set(), []
    for d in dates:
        g = pred.loc[d]
        if not isinstance(g, pd.DataFrame) or len(g) < 2 * k:
            continue
        top = g.nlargest(k, col)
        names = set(top.stock)
        new_frac = len(names - prev) / k                       # share of the portfolio that was replaced
        gross = top.fwd_ret.mean()
        net = gross - cost * new_frac
        rows.append((d, gross, net, g.fwd_ret.mean(), g.nifty_fwd_ret.iloc[0], new_frac))
        prev = names
    return pd.DataFrame(rows, columns=["date", "gross", "net", "ew", "nifty", "turnover"]).set_index("date")
 
 
def growth_stats(r, h):
    eq = (1 + r).cumprod()
    ppy = 252 / h
    return 100 * (eq.iloc[-1] ** (ppy / len(r)) - 1), 100 * (eq / eq.cummax() - 1).min()
 
 
def report(pred, h):
    print(f"\nStocks: {pred.stock.nunique()}   rows: {len(pred)}   share label = 1: {100 * pred.y.mean():.1f}%")
 
    print(f"\n=== AUC by test year, {h}-day relative label (0.50 = no skill) ===")
    print(f"{'Year':<6}{'Rows':>8}" + "".join(f"{n[:19]:>21}" for n, _ in SCORES))
    for year, g in pred.groupby(pred.index.year):
        print(f"{year:<6}{len(g):>8}" + "".join(f"{safe_auc(g.y, g[c]):>21.3f}" for _, c in SCORES))
    print(f"{'ALL':<6}{len(pred):>8}" + "".join(f"{safe_auc(pred.y, pred[c]):>21.3f}" for _, c in SCORES))
    for n, c in SCORES[:3]:
        py = pred.groupby(pred.index.year).apply(lambda g: safe_auc(g.y, g[c]))
        print(f"   {n}: years with AUC above 0.53: {int((py > 0.53).sum())} of {len(py)}")
 
    print("\n=== Information coefficient (rank correlation of score with realised excess return, per day) ===")
    print(f"{'Model':<22}{'MeanIC':>9}{'IC>0 days':>11}{'IC/std':>9}   mean IC by year")
    for n, c in SCORES:
        ic = daily_ic(pred, c)
        by_year = ic.groupby(ic.index.year).mean()
        print(f"{n:<22}{ic.mean():>9.4f}{100 * (ic > 0).mean():>10.1f}%{ic.mean() / ic.std():>9.3f}   "
              + " ".join(f"{y}:{v:+.3f}" for y, v in by_year.items()))
    print("   (an IC of about 0.02-0.05 is already considered useful in practice; days overlap, so be careful)")
 
    print(f"\n=== Top-{K} portfolio, rebalanced every {h} trading days, {100 * COST:.2f}% cost on each replaced stock ===")
    print("Compared with the equal-weight universe of the same stocks (no costs for the benchmark) and with NIFTY.")
    verdicts = {}
    for n, c in SCORES[:3]:
        bt = topk_backtest(pred, c, h)
        if bt.empty:
            continue
        excess = bt.net - bt.ew
        by_year = excess.groupby(excess.index.year).mean()
        ann, dd = growth_stats(bt.net, h)
        ann_ew, dd_ew = growth_stats(bt.ew, h)
        ann_n, dd_n = growth_stats(bt.nifty, h)
        print(f"\n{n}:  {len(bt)} periods, average turnover {100 * bt.turnover.mean():.0f}% of the portfolio per period")
        print(f"   average per period:  top-{K} gross {100 * bt.gross.mean():+.3f}%   net {100 * bt.net.mean():+.3f}%   "
              f"universe {100 * bt.ew.mean():+.3f}%   NIFTY {100 * bt.nifty.mean():+.3f}%")
        print(f"   net excess over the universe: {100 * excess.mean():+.3f}% per period, "
              f"beat it in {100 * (excess > 0).mean():.0f}% of periods")
        print(f"   by year (net excess, % per period): " + " ".join(f"{y}:{100 * v:+.2f}" for y, v in by_year.items()))
        print(f"   yearly growth rate / worst drop:  portfolio {ann:+.1f}% / {dd:.1f}%   "
              f"universe {ann_ew:+.1f}% / {dd_ew:.1f}%   NIFTY {ann_n:+.1f}% / {dd_n:.1f}%")
        verdicts[n] = (100 * excess.mean(), int((by_year > 0).sum()), len(by_year))
 
    print("\nBar set in advance: AUC above 0.53 in at least 5 of 8 years, AND the top-10 portfolio beats the "
          "equal-weight universe by at least +0.30% per period after costs, in at least 5 of 8 years.")
    for n, (avg, pos, tot) in verdicts.items():
        ok = avg >= 0.30 and pos >= 5
        print(f"   {n}: net excess {avg:+.3f}% per period, positive in {pos} of {tot} years -> "
              f"{'portfolio part MET' if ok else 'portfolio part NOT met'}")
    print("Note: picking the best of three models after seeing the results would inflate the outcome.")
 
 
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("5", "10", "20"):
        print("Use: python src/v2.py 20   or   python src/v2.py 5")
        return
    h = int(sys.argv[1])
    nifty_path = RAW_DIR / "NIFTY.parquet"
    if not nifty_path.exists():
        print("data/raw/NIFTY.parquet not found. Run python src/data_index.py first.")
        return
    nifty_close = pd.read_parquet(nifty_path).Close
    names = sorted(p.name.replace("_dataset_rel.parquet", "") for p in PROC_DIR.glob("*_dataset_rel.parquet"))
    dev = {}
    for n in names:
        raw_path = RAW_DIR / f"{n}.parquet"
        if raw_path.exists():
            dev[n] = build_stock(pd.read_parquet(raw_path), nifty_close, h)
        else:
            print(f"{n}: {raw_path} not found, skipped")
    if not dev:
        print("No stock data found. Run build_universe.py first.")
        return
    frames, use = prepare(dev, nifty_close)
    print(f"Version 2: {len(frames)} stocks, {len(use)} features, {h}-day relative label")
    pred = walk_forward_v2(frames, use, h)
    if pred.empty:
        return
    RES_DIR.mkdir(exist_ok=True)
    out = RES_DIR / f"walk_forward_predictions_v2_h{h}.csv"
    pred.to_csv(out)
    print(f"Test period: {pred.index.min().date()} to {pred.index.max().date()} "
          "(locked final period NOT used)")
    report(pred, h)
    print(f"\nPredictions saved to {out}")
 
 
if __name__ == "__main__":
    main()