"""
Paper trading for the Version 2 models on NSE large caps. No real money, no orders: it only records and scores picks.

Run from the project folder (venv active, internet needed):
    python src/paper.py signal     every 20 trading days: refresh data, retrain, record the top-10 picks per model
    python src/paper.py score      any time: score every finished holding period from the ledger

Rules (fixed in advance, do not change them after seeing results):
  * Models: Logistic Regression, XGBoost classifier, XGBoost ranker (same settings as Version 2), trained on ALL rows whose
    20-day label is already known on the signal day. Same 31 features as Experiment 6.
  * Picks: top 10 of the 45 stocks per model. Rebalance every 20 trading days. Benchmark: all 45 stocks equal weight.
  * Entry at the first trading day's open AFTER the pick was recorded (and after the signal day); exit at the open 20 trading days later. Cost 0.30% on every replaced stock.
  * The dashboard (streamlit run src/dashboard.py, Paper trading tab) runs the same two steps with buttons.
  * Judge only after at least 24 periods (about two years), with the same bar as before: net excess of at least +0.30% a month
    and a 95% interval that excludes zero. Fewer periods cannot tell skill from luck.
Files: paper/ledger.csv (picks), paper/score.csv (results).
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import v2
from features import make_features
from horizon import make_label_h

H, K, COST = 20, 10, 0.003
PAPER = Path("paper")
LEDGER, SCORE = PAPER / "ledger.csv", PAPER / "score.csv"
MODELS = [("Logistic Regression", "s_logit"), ("XGBoost classifier", "s_xgb"), ("XGBoost ranker", "s_rank")]


def stock_names():
    return sorted(p.stem for p in Path("data/raw").glob("*.parquet") if p.stem != "NIFTY")


IST = timezone(timedelta(hours=5, minutes=30))


def drop_unfinished_day(df, now=None):
    """Before 4 pm India time, today's candle is unfinished (or missing): never use it as a close."""
    now = now or datetime.now(IST)
    if now.hour < 16 and len(df) and df.index[-1].date() >= now.date():
        return df[df.index.date < now.date()]
    return df


def download_fresh(names):
    import yfinance as yf

    def get(ticker):
        df = yf.download(ticker, start="2012-01-01", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return drop_unfinished_day(df[["Open", "High", "Low", "Close", "Volume"]].dropna())
    raws = {}
    for n in names:
        try:
            raws[n] = get(n.replace("_", "&") + ".NS")
        except Exception as e:
            print(f"  {n}: skipped ({e})")
    return raws, get("^NSEI").Close


def live_frame(raws, nifty):
    """Same features as Version 2, kept for ALL dates (labels are NaN for the newest rows)."""
    parts = {}
    for n, raw in raws.items():
        feats = pd.concat([make_features(raw), v2.long_features(raw)], axis=1)
        y, s, nf, ex = make_label_h(raw, nifty, H)
        parts[n] = (feats.join(y.rename("y")).join(s.rename("fwd_ret"))
                         .join(nf.rename("nifty_fwd_ret")).join(ex.rename("excess_ret")))
    big = pd.concat([d.assign(stock=n) for n, d in parts.items()])
    stock_cols = [c for c in big.columns if c not in v2.LABEL_COLS + ["stock"]]
    big = big.dropna(subset=stock_cols)
    big["breadth50"] = (big.dist_sma50 > 0).groupby(level=0).transform("mean")
    big["breadth_ret20"] = (big.ret_20 > 0).groupby(level=0).transform("mean")
    rk = big.groupby(level=0)[stock_cols].rank(pct=True)
    for c in stock_cols:
        big["r_" + c] = rk[c].values
    mk = v2.market_feats(nifty)
    mk = mk.reindex(mk.index.union(big.index.unique())).ffill().reindex(big.index)
    for c in mk.columns:
        big[c] = mk[c].values
    use = ["r_" + c for c in stock_cols] + list(mk.columns) + ["breadth50", "breadth_ret20"]
    return big, use


def make_signal(raws, nifty):
    big, use = live_frame(raws, nifty)
    counts = big.groupby(level=0).stock.count()
    signal_date = counts[counts >= 0.8 * len(raws)].index.max()
    live = big.loc[signal_date].dropna(subset=use)
    train = big.dropna(subset=v2.LABEL_COLS + use)
    train = train[train.index < signal_date]
    scores = pd.DataFrame({"stock": live.stock.values}, index=live.index)
    for name, model in v2.make_classifiers().items():
        model.fit(train[use], train.y)
        scores["s_" + name] = model.predict_proba(live[use])[:, 1]
    tr = train.sort_index(kind="stable")
    qid = pd.factorize(tr.index)[0]
    rel = np.minimum((tr.groupby(level=0).excess_ret.rank(pct=True).values * 5).astype(int), 4)
    ranker = v2.make_ranker()
    ranker.fit(tr[use], rel, qid=qid)
    scores["s_rank"] = ranker.predict(live[use])
    scores["vol_rank"] = live["r_vol60"].values
    scores["mom_rank"] = live["r_ret_20"].values
    rows = []
    for _, c in MODELS:
        top = scores.nlargest(K, c)
        for rank, (st, sc, vr, mr) in enumerate(zip(top.stock, top[c], top.vol_rank, top.mom_rank), 1):
            rows.append((signal_date, c, st, rank, sc, vr, mr))
    return signal_date, pd.DataFrame(rows, columns=["signal_date", "model", "stock", "rank", "score", "vol_rank", "mom_rank"])


def run_signal():
    """Returns ("not_due", days_left) or ("done", (signal_date, picks))."""
    PAPER.mkdir(exist_ok=True)
    raws, nifty = download_fresh(stock_names())
    if LEDGER.exists():
        led = pd.read_csv(LEDGER, parse_dates=["signal_date"])
        since = int((nifty.index > led.signal_date.max()).sum())
        if since < H:
            return "not_due", H - since
    date, picks = make_signal(raws, nifty)
    picks["created"] = datetime.now(IST).date().isoformat()     # entry can only happen AFTER this day
    led = pd.concat([led, picks]) if LEDGER.exists() else picks
    led.to_csv(LEDGER, index=False)
    return "done", (date, picks)


def cmd_signal():
    print("Refreshing data ...")
    status, res = run_signal()
    if status == "not_due":
        print(f"Not due yet: {res} more trading days.")
        return
    date, picks = res
    print(f"Signal date {date.date()}. Buy at the NEXT trading day's open. Picks saved to {LEDGER}\n")
    for name, col in MODELS:
        print(f"{name}: " + ", ".join(picks[picks.model == col].stock))


def score_ledger(led, raws):
    rows, detail, prev = [], [], {}
    for date, g in led.groupby("signal_date"):
        ret = {}
        eff = date
        if "created" in g.columns and g.created.notna().any():
            eff = max(date, pd.Timestamp(g.created.dropna().iloc[0]))   # cannot buy before the pick was recorded
        for n, raw in raws.items():
            pos = raw.index.searchsorted(eff, side="right")          # first trading day after the later of the two dates
            if pos + H < len(raw):
                ret[n] = raw.Open.iloc[pos + H] / raw.Open.iloc[pos] - 1
        if len(ret) < 0.8 * len(raws):
            continue                                                 # period not finished yet
        ew = np.mean(list(ret.values()))
        row = {"signal_date": date.date(), "universe": ew}
        for _, col in MODELS:
            gm = g[g.model == col]
            names = [s for s in gm.stock if s in ret]
            for _, r in gm[gm.stock.isin(names)].iterrows():
                detail.append({"signal_date": date.date(), "model": col, "stock": r.stock, "rank": r.get("rank"),
                               "ret": ret[r.stock], "excess": ret[r.stock] - ew,
                               "vol_rank": r.get("vol_rank"), "mom_rank": r.get("mom_rank")})
            gross = np.mean([ret[s] for s in names])
            new_frac = len(set(names) - prev.get(col, set())) / K
            row[col] = gross - COST * new_frac - ew
            prev[col] = set(names)
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(detail)


def run_score():
    """Scores finished periods, saves paper/score.csv and paper/detail.csv. Returns (score, detail) or None."""
    if not LEDGER.exists():
        return None
    led = pd.read_csv(LEDGER, parse_dates=["signal_date"])
    raws, _ = download_fresh(stock_names())
    sc, detail = score_ledger(led, raws)
    if sc.empty:
        return None
    sc.to_csv(SCORE, index=False)
    detail.to_csv(PAPER / "detail.csv", index=False)
    return sc, detail


def cmd_score():
    res = run_score()
    if res is None:
        print("Nothing to score yet (no ledger, or no holding period has finished: each lasts 20 trading days after the next open).")
        return
    sc, _ = res
    print(sc.round(4).to_string(index=False))
    n = len(sc)
    print(f"\nFinished periods: {n}  (need at least 24 before judging)")
    for name, col in MODELS:
        x = sc[col]
        se = x.std() / np.sqrt(n) if n > 1 else float("nan")
        print(f"{name}: net excess vs universe {100 * x.mean():+.3f}% per period, standard error {100 * se:.3f}%, "
              f"beat the universe in {100 * (x > 0).mean():.0f}% of periods")
    if n < 24:
        print("Too few periods to say anything. A good or bad run now is mostly luck.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"signal": cmd_signal, "score": cmd_score}.get(cmd, lambda: print("Use: python src/paper.py signal | score"))()