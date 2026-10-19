from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from features import ATR_N, atr
 
STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
RAW_DIR = Path("data/raw")
RES_DIR = Path("results")
 
PROB_THRESH = 0.55
HORIZON = 5
STOP_ATR = 1.5
TARGET_ATR = 3.0
RISK_PER_TRADE = 0.01
MAX_POSITION = 0.20
COST = 0.003
COST_GRID = (0.002, 0.003, 0.004)
 
 
def simulate(raw, signal_dates):
    """Simulate trades for one stock. Returns a list of trade dicts (gross returns, before costs)."""
    a = atr(raw, ATR_N)
    idx = raw.index
    n = len(raw)
    o_, h_, l_, c_ = (raw.Open.values, raw.High.values, raw.Low.values, raw.Close.values)
    trades, next_free = [], 0
    for d in signal_dates:
        i = idx.get_loc(d)
        if i < next_free or i + 1 >= n or np.isnan(a.iloc[i]):
            continue
        entry = o_[i + 1]
        atr_i = a.iloc[i]
        stop = entry - STOP_ATR * atr_i
        target = entry + TARGET_ATR * atr_i
        last = min(i + HORIZON, n - 1)
        exit_px, reason, j = None, "time", last
        for j in range(i + 1, last + 1):
            if o_[j] <= stop:
                exit_px, reason = o_[j], "gap_stop"; break
            if o_[j] >= target:
                exit_px, reason = o_[j], "gap_target"; break
            if l_[j] <= stop:
                exit_px, reason = stop, "stop"; break          # stop first if both hit
            if h_[j] >= target:
                exit_px, reason = target, "target"; break
        if exit_px is None:
            j, exit_px, reason = last, c_[last], "time"
        stop_pct = (entry - stop) / entry
        frac = min(RISK_PER_TRADE / stop_pct, MAX_POSITION) if stop_pct > 0 else 0.0
        trades.append({"signal_date": d, "entry_date": idx[i + 1], "exit_date": idx[j],
                       "entry": entry, "exit": exit_px, "reason": reason,
                       "gross": exit_px / entry - 1, "frac": frac, "days": j - i})
        next_free = j + 1
    return trades
 
 
def run_strategy(pred, raws, col, thresh):
    """col=None means 'always long' (a signal on every test day)."""
    out = []
    for st in STOCKS:
        if st not in raws:
            continue
        p = pred[pred.stock == st]
        dates = p.index if col is None else p.index[p[col].values >= thresh]
        out += [dict(t, stock=st) for t in simulate(raws[st], dates)]
    return pd.DataFrame(out)
 
 
def metrics(trades, cost):
    if trades.empty:
        return None
    net = trades.gross - cost
    pnl = (trades.frac * net).groupby(trades.exit_date).sum().sort_index()
    equity = pd.concat([pd.Series([1.0]), 1 + pnl.cumsum()]).reset_index(drop=True)
    return {"trades": len(trades), "win": 100 * (net > 0).mean(),
            "avg_net": 100 * net.mean(), "avg_gross": 100 * trades.gross.mean(),
            "total": 100 * pnl.sum(), "maxdd": 100 * (equity - equity.cummax()).min(),
            "days": trades.days.mean()}
 
 
def print_table(strategies, cost):
    print(f"{'Strategy':<22}{'Trades':>7}{'Win%':>7}{'AvgNet%':>9}{'Total%':>9}{'MaxDD%':>8}{'AvgDays':>8}")
    for name, tr in strategies.items():
        m = metrics(tr, cost)
        if m is None:
            print(f"{name:<22} no trades")
            continue
        print(f"{name:<22}{m['trades']:>7}{m['win']:>7.1f}{m['avg_net']:>9.2f}"
              f"{m['total']:>9.1f}{m['maxdd']:>8.1f}{m['days']:>8.1f}")
 
 
def buy_and_hold(pred, raws):
    rets = []
    for st in STOCKS:
        if st not in raws:
            continue
        p = pred[pred.stock == st]
        c = raws[st].Close.loc[p.index.min():p.index.max()]
        rets.append(c.iloc[-1] / c.iloc[0] - 1)
    return 100 * np.mean(rets)
 
 
def main():
    path = RES_DIR / "walk_forward_predictions.csv"
    if not path.exists():
        print(f"{path} not found. Run model.py first.")
        return
    pred = pd.read_csv(path, index_col=0, parse_dates=True)
    raws = {st: pd.read_parquet(RAW_DIR / f"{st}.parquet") for st in STOCKS
            if (RAW_DIR / f"{st}.parquet").exists()}
 
    strategies = {
        "Random Forest >= 0.55": run_strategy(pred, raws, "prob_rf", PROB_THRESH),
        "XGBoost >= 0.55": run_strategy(pred, raws, "prob_xgb", PROB_THRESH),
        "Always long (baseline)": run_strategy(pred, raws, None, 0),
    }
    for name, tr in strategies.items():
        if not tr.empty:
            tr.to_csv(RES_DIR / f"trades_{name.split()[0].lower()}.csv", index=False)
 
    print(f"Test period: {pred.index.min().date()} to {pred.index.max().date()}")
    print(f"\n=== Results with {100 * COST:.2f}% round-trip cost ===")
    print_table(strategies, COST)
    print(f"{'Buy & hold (avg of 5)':<22} {buy_and_hold(pred, raws):>8.1f}% per stock over the whole period "
          "(fully invested, so not directly comparable to Total%)")
 
    print("\n=== Average net return per trade (%) at different costs ===")
    print(f"{'Strategy':<22}" + "".join(f"{100 * c:>9.2f}%" for c in COST_GRID))
    for name, tr in strategies.items():
        if not tr.empty:
            print(f"{name:<22}" + "".join(f"{100 * (tr.gross - c).mean():>10.2f}" for c in COST_GRID))
 
    print("\n=== By year: trades / average net return per trade (%) ===")
    names = list(strategies)
    print(f"{'Year':<6}" + "".join(f"{n[:18]:>22}" for n in names))
    years = sorted({y for tr in strategies.values() if not tr.empty for y in tr.exit_date.dt.year.unique()})
    for y in years:
        row = f"{y:<6}"
        for n in names:
            tr = strategies[n]
            g = tr[tr.exit_date.dt.year == y] if not tr.empty else tr
            row += f"{len(g):>9d} / {100 * (g.gross - COST).mean():>6.2f}  " if len(g) else f"{'-':>22}"
        print(row)
 
    print("\n=== Why trades ended (share of trades) ===")
    for name, tr in strategies.items():
        if not tr.empty:
            share = (100 * tr.reason.value_counts(normalize=True)).round(1).to_dict()
            print(f"{name:<22} {share}")
 
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, tr in strategies.items():
        if tr.empty:
            continue
        pnl = (tr.frac * (tr.gross - COST)).groupby(tr.exit_date).sum().sort_index()
        ax.plot(pnl.index, 100 * pnl.cumsum(), label=name, lw=1.2)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"Cumulative return on capital (non-compounded, {100 * COST:.2f}% cost)")
    ax.set_ylabel("% of capital")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES_DIR / "equity_week3.png", dpi=110)
    print(f"\nTrade logs and chart saved in '{RES_DIR}' (equity_week3.png).")
 
 
if __name__ == "__main__":
    main()