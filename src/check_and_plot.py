from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")  # save to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from features import make_features
from labels import HORIZON, UP_THRESH
 
STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
RES_DIR = Path("results")
N_SAMPLES = 40
 
 
def leakage_test(raw, n_samples=N_SAMPLES, seed=0):
    """Recompute features using ONLY rows up to day t and compare with the full-data value."""
    full = make_features(raw)
    rng = np.random.default_rng(seed)
    days = rng.choice(np.arange(120, len(raw) - 1), size=n_samples, replace=False)
    bad = 0
    for t in days:
        cut = make_features(raw.iloc[: t + 1]).iloc[-1]
        if not np.allclose(full.iloc[t].values, cut.values, equal_nan=True, rtol=1e-9, atol=1e-12):
            bad += 1
    return bad, len(days)
 
 
def label_test(raw, dataset, n_samples=N_SAMPLES, seed=1):
    """Check labels against a manual calculation from raw closes."""
    rng = np.random.default_rng(seed)
    pos = rng.choice(len(dataset), size=n_samples, replace=False)
    bad = 0
    for p in pos:
        d = dataset.index[p]
        i = raw.index.get_loc(d)
        manual = raw.Close.iloc[i + HORIZON] / raw.Close.iloc[i] - 1
        if float(manual > UP_THRESH) != dataset.y.iloc[p]:
            bad += 1
    return bad, len(pos)
 
 
def correlation_test(dataset):
    """Correlation of each feature with the label and the forward return."""
    feats = dataset.drop(columns=["y", "fwd_ret"])
    out = pd.DataFrame({
        "corr_with_label": feats.corrwith(dataset.y),
        "corr_with_fwd_ret": feats.corrwith(dataset.fwd_ret),
    })
    return out.reindex(out.corr_with_label.abs().sort_values(ascending=False).index)
 
 
def plot_stock(name, raw, dataset):
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(raw.index, raw.Close, lw=0.9)
    ax[0].set_title(f"{name}: adjusted close")
    ax[1].plot(dataset.index, dataset.rsi14, lw=0.6)
    ax[1].axhline(70, ls="--", lw=0.7)
    ax[1].axhline(30, ls="--", lw=0.7)
    ax[1].set_title("RSI(14), should stay between 0 and 100")
    ax[2].plot(dataset.index, dataset.vol20, lw=0.7)
    ax[2].set_title("20-day volatility (spikes in 2020 are expected)")
    yearly = dataset.y.rolling(250).mean()
    ax[3].plot(dataset.index, yearly, lw=0.9)
    ax[3].axhline(0.5, ls="--", lw=0.7)
    ax[3].set_title("Share of label = 1 over the past ~year")
    fig.tight_layout()
    fig.savefig(RES_DIR / f"sanity_{name}.png", dpi=110)
    plt.close(fig)
 
 
def main():
    RES_DIR.mkdir(exist_ok=True)
    all_corr = []
    problems = 0
    for name in STOCKS:
        raw_path = RAW_DIR / f"{name}.parquet"
        ds_path = PROC_DIR / f"{name}_dataset.parquet"
        if not raw_path.exists() or not ds_path.exists():
            print(f"{name}: files missing. Run data.py, features.py and labels.py first.")
            continue
        raw = pd.read_parquet(raw_path)
        ds = pd.read_parquet(ds_path)
 
        bad_f, n_f = leakage_test(raw)
        bad_l, n_l = label_test(raw, ds)
        problems += bad_f + bad_l
        print(f"{name:<10} leakage test: {n_f - bad_f}/{n_f} passed   "
              f"label test: {n_l - bad_l}/{n_l} passed   "
              f"RSI range: {ds.rsi14.min():.1f} to {ds.rsi14.max():.1f}")
        all_corr.append(correlation_test(ds))
        plot_stock(name, raw, ds)
 
    if all_corr:
        avg = sum(all_corr) / len(all_corr)
        avg = avg.reindex(avg.corr_with_label.abs().sort_values(ascending=False).index)
        print("\nAverage correlation of each feature with the label (top 8):")
        print(avg.head(8).round(3).to_string())
        top = avg.corr_with_label.abs().max()
        if top > 0.15:
            print("\nWARNING: a feature correlates strongly with the label. Look for leakage.")
        else:
            print(f"\nStrongest correlation is {top:.3f}. Small values are normal: "
                  "daily stock features are only weakly related to the future.")
 
    print("\nRESULT:", "all checks passed" if problems == 0 else f"{problems} checks FAILED, investigate before continuing")
    print(f"Charts saved in the '{RES_DIR}' folder.")
 
 
if __name__ == "__main__":
    main()