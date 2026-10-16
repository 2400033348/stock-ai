"""
V1 baseline: walk-forward Random Forest + ATR stop/target backtest with costs.

Install:  pip install yfinance pandas numpy scikit-learn
Run:      python v1_baseline.py

Research tool only. Not investment advice.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ---------------- settings ----------------
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
START = "2012-01-01"
DEV_END = "2025-10-14"          # last 12 months stay locked during development
RUN_LOCKED = False              # set True ONLY once, at the very end
FIRST_TEST_YEAR = 2018

HORIZON = 5                     # trading days
UP_THRESH = 0.005               # label = 1 if 5-day return > 0.5%
PROB_THRESH = 0.55
ATR_N = 14
STOP_ATR = 1.5
TARGET_ATR = 3.0
COST_ROUND_TRIP = 0.003         # 0.30% round trip, try 0.002 and 0.004 too
RISK_PER_TRADE = 0.01
MAX_POSITION = 0.20
SEED = 42


# ---------------- data ----------------
def load(ticker):
    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[df.Volume > 0]
    if not RUN_LOCKED:
        df = df[df.index <= DEV_END]
    return df


# ---------------- features ----------------
def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df, n=ATR_N):
    pc = df.Close.shift(1)
    tr = pd.concat([df.High - df.Low,
                    (df.High - pc).abs(),
                    (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def make_features(df):
    c = df.Close
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 5, 10, 20):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (10, 20, 50):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["rsi14"] = rsi(c)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    f["macd"] = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    f["vol20"] = c.pct_change().rolling(20).std()
    f["atr_pct"] = atr(df) / c
    f["vol_ratio"] = df.Volume / df.Volume.rolling(20).mean()
    f["range_pct"] = (df.High - df.Low) / c
    f["gap"] = df.Open / c.shift(1) - 1
    return f


def make_label(df):
    fwd = df.Close.shift(-HORIZON) / df.Close - 1
    y = (fwd > UP_THRESH).astype(float)
    y[fwd.isna()] = np.nan          # last HORIZON rows have no label
    return y


# ---------------- walk-forward model ----------------
def walk_forward(f, y):
    data = f.join(y.rename("y")).dropna()
    probs = []
    last_year = data.index.year.max()
    for year in range(FIRST_TEST_YEAR, last_year + 1):
        test = data[data.index.year == year]
        train = data[data.index < f"{year}-01-01"]
        train = train.iloc[:-HORIZON]            # embargo: labels overlap test period
        if len(train) < 500 or test.empty:
            continue
        model = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            random_state=SEED, n_jobs=-1)
        model.fit(train.drop(columns="y"), train.y)
        p = model.predict_proba(test.drop(columns="y"))[:, 1]
        probs.append(pd.DataFrame({"prob": p, "y": test.y}, index=test.index))
    return pd.concat(probs) if probs else pd.DataFrame(columns=["prob", "y"])


# ---------------- backtest ----------------
def backtest(df, prob, thresh):
    a = atr(df)
    idx = df.index
    n = len(df)
    trades, next_free = [], 0
    for d, p in prob.items():
        i = idx.get_loc(d)
        if p < thresh or i < next_free or i + 1 >= n or np.isnan(a.iloc[i]):
            continue
        entry = df.Open.iloc[i + 1]                      # enter at next open
        stop = entry - STOP_ATR * a.iloc[i]
        target = entry + TARGET_ATR * a.iloc[i]
        last = min(i + HORIZON, n - 1)
        exit_px, j = None, last
        for j in range(i + 1, last + 1):
            o, h, l = df.Open.iloc[j], df.High.iloc[j], df.Low.iloc[j]
            if o <= stop:
                exit_px = o; break                       # gap through stop
            if l <= stop:
                exit_px = stop; break                    # stop first if both hit
            if h >= target:
                exit_px = target; break
        if exit_px is None:
            j = last
            exit_px = df.Close.iloc[j]                   # time exit
        net = exit_px / entry - 1 - COST_ROUND_TRIP
        stop_pct = (entry - stop) / entry
        frac = min(RISK_PER_TRADE / stop_pct, MAX_POSITION) if stop_pct > 0 else 0
        trades.append({"exit_date": idx[j], "ret": net, "pnl": frac * net})
        next_free = j + 1                                # one position at a time
    return trades


def summarize(name, trades):
    if not trades:
        print(f"{name:<22} no trades")
        return
    t = pd.DataFrame(trades)
    pnl = t.groupby("exit_date").pnl.sum().sort_index()
    equity = 1 + pnl.cumsum()                            # simple, non-compounded
    dd = (equity - equity.cummax()).min()
    print(f"{name:<22} trades={len(t):4d}  win={100*(t.ret>0).mean():5.1f}%  "
          f"avg_net={100*t.ret.mean():6.2f}%  total={100*pnl.sum():7.1f}%  "
          f"maxDD={100*dd:6.1f}%")


# ---------------- main ----------------
def main():
    all_model, all_base, all_pred, bh = [], [], [], []
    for tk in TICKERS:
        df = load(tk)
        if len(df) < 800:
            print(f"skip {tk}: not enough data")
            continue
        f, y = make_features(df), make_label(df)
        pred = walk_forward(f, y)
        if pred.empty:
            continue
        all_pred.append(pred)
        all_model += backtest(df, pred.prob, PROB_THRESH)
        all_base += backtest(df, pred.prob, 0.0)         # always-long baseline
        span = df.loc[pred.index[0]:pred.index[-1], "Close"]
        bh.append(span.iloc[-1] / span.iloc[0] - 1)

    pooled = pd.concat(all_pred)
    print(f"\nPooled walk-forward AUC: {roc_auc_score(pooled.y, pooled.prob):.3f}"
          f"  (0.50 = no skill)")
    print(f"Cost per round trip: {100*COST_ROUND_TRIP:.2f}%   "
          f"Prob threshold: {PROB_THRESH}\n")
    summarize("Model signals", all_model)
    summarize("Always-long baseline", all_base)
    print(f"{'Buy & hold (avg)':<22} total={100*np.mean(bh):.1f}% per stock over test span")


if __name__ == "__main__":
    main()
