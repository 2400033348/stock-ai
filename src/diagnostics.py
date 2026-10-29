
"""
Experiment 7 (diagnostics): is the Version 2 result distinguishable from luck? No retraining.
 
Run from the project folder:
    python src/diagnostics.py
 
Reads results/walk_forward_predictions_v2_h20.csv (test years 2018-2025 only; the locked period is not touched).
For each model it reports, with 95% block-bootstrap confidence intervals:
  * mean information coefficient (IC) per day,
  * top-10 net excess return over the equal-weight universe per month,
  * annualised Sharpe ratios (portfolio, universe, and the excess = information ratio),
  * worst drop inside each year.
If an interval contains zero, the result cannot be told apart from chance.
Writes results/diagnostics_v2_h20.txt
"""
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from v2 import daily_ic, topk_backtest
 
H = 20
SCORES = [("Logistic Regression", "s_logit"), ("XGBoost classifier", "s_xgb"), ("XGBoost ranker", "s_rank")]
N_BOOT, SEED = 10000, 42
 
 
def block_bootstrap_mean(x, block, n_boot=N_BOOT, seed=SEED):
    """95% interval for the mean of a serially correlated series (resample blocks of consecutive values)."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])
 
 
def sharpe(r, ppy):
    return np.sqrt(ppy) * r.mean() / r.std() if r.std() > 0 else np.nan
 
 
def yearly_drawdown(r):
    out = {}
    for y, g in r.groupby(r.index.year):
        eq = (1 + g).cumprod()
        out[y] = 100 * (eq / eq.cummax() - 1).min()
    return out
 
 
def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/walk_forward_predictions_v2_h20.csv")
    if not path.exists():
        print(f"{path} not found. Run python src/v2.py 20 first.")
        return
    pred = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    lines = []
 
    def say(s=""):
        print(s)
        lines.append(s)
 
    ppy = 252 / H
    say(f"Diagnostics for Version 2, {H}-day label. Test period {pred.index.min().date()} to "
        f"{pred.index.max().date()}, {pred.stock.nunique()} stocks.")
    say("95% intervals come from a block bootstrap (blocks of consecutive values), so serial correlation is respected.")
    say("If an interval contains zero, the result cannot be told apart from chance.")
    for name, col in SCORES:
        say(f"\n=== {name} ===")
        ic = daily_ic(pred, col)
        lo, hi = block_bootstrap_mean(ic.values, block=H)
        say(f"Mean IC per day: {ic.mean():+.4f}   95% interval [{lo:+.4f}, {hi:+.4f}]   "
            f"-> {'excludes zero' if lo > 0 or hi < 0 else 'INCLUDES zero'}")
        bt = topk_backtest(pred, col, H)
        excess = bt.net - bt.ew
        lo, hi = block_bootstrap_mean(excess.values, block=3)
        say(f"Net excess vs universe per month: {100 * excess.mean():+.3f}%   95% interval "
            f"[{100 * lo:+.3f}%, {100 * hi:+.3f}%]   -> {'excludes zero' if lo > 0 or hi < 0 else 'INCLUDES zero'}")
        gross_ex = bt.gross - bt.ew
        say(f"   gross excess {100 * gross_ex.mean():+.3f}%   turnover cost {100 * (bt.gross - bt.net).mean():.3f}%   "
            f"net excess {100 * excess.mean():+.3f}%   (per month)")
        say(f"Sharpe (annualised):  portfolio {sharpe(bt.net, ppy):.2f}   universe {sharpe(bt.ew, ppy):.2f}   "
            f"NIFTY {sharpe(bt.nifty, ppy):.2f}   excess (information ratio) {sharpe(excess, ppy):+.2f}")
        dd = yearly_drawdown(bt.net)
        dd_u = yearly_drawdown(bt.ew)
        say("Worst drop inside each year (portfolio / universe, %):  "
            + "  ".join(f"{y}: {dd[y]:.1f}/{dd_u[y]:.1f}" for y in dd))
    say("\nReading guide: an information ratio below about 0.5 with an interval that includes zero is not evidence "
        "of a tradable edge. The universe is made of today's surviving large caps, so its returns are flattering.")
    Path("results").mkdir(exist_ok=True)
    Path("results/diagnostics_" + path.stem.replace("walk_forward_predictions_", "") + ".txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nDiagnostics saved in the results folder.")
 
 
if __name__ == "__main__":
    main()