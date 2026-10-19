import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
 
STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
PROC_DIR = Path("data/processed")
RES_DIR = Path("results")
 
DEV_END = "2025-10-14"     # everything after this date is the locked final test
FIRST_TEST_YEAR = 2018
HORIZON = 5                # must match labels.py
SEED = 42
MIN_TRAIN_ROWS = 500
NON_FEATURES = ["y", "fwd_ret", "excess_ret", "nifty_fwd_ret"]
 
 
def make_models():
    return {
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            random_state=SEED, n_jobs=-1),
        "xgb": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            random_state=SEED, n_jobs=-1, eval_metric="logloss"),
    }
 
 
def dev_data(ds):
    """Keep only development rows. Drop the last HORIZON rows so no label peeks past DEV_END."""
    dev = ds[ds.index <= DEV_END]
    return dev.iloc[:-HORIZON]
 
 
def walk_forward(ds, first_year=FIRST_TEST_YEAR):
    """Return a DataFrame of out-of-sample predictions for one stock."""
    dev = dev_data(ds)
    feature_cols = [c for c in dev.columns if c not in NON_FEATURES]
    parts = []
    for year in range(first_year, dev.index.year.max() + 1):
        test = dev[dev.index.year == year]
        train = dev[dev.index < f"{year}-01-01"].iloc[:-HORIZON]   # embargo
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        out = pd.DataFrame({"y": test.y, "fwd_ret": test.fwd_ret,
                            "mom5": test.ret_5}, index=test.index)
        if "excess_ret" in test.columns:
            out["excess_ret"] = test.excess_ret
        for name, model in make_models().items():
            model.fit(train[feature_cols], train.y)
            out[f"prob_{name}"] = model.predict_proba(test[feature_cols])[:, 1]
        parts.append(out)
    return pd.concat(parts) if parts else pd.DataFrame()
 
 
def safe_auc(y, score):
    return roc_auc_score(y, score) if y.nunique() == 2 else np.nan
 
 
def calibration_table(y, prob, bins=(0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.01)):
    cut = pd.cut(prob, bins=list(bins), right=False)
    t = pd.DataFrame({"y": y, "p": prob, "bin": cut}).groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return t
 
 
def report(pred):
    models = [("Random Forest", "prob_rf"), ("XGBoost", "prob_xgb"),
              ("Momentum (ret_5)", "mom5")]
 
    print("\n=== AUC by test year (all stocks pooled; 0.50 = no skill) ===")
    print(f"{'Year':<6}{'Rows':>7}" + "".join(f"{n:>20}" for n, _ in models))
    for year, g in pred.groupby(pred.index.year):
        print(f"{year:<6}{len(g):>7}" +
              "".join(f"{safe_auc(g.y, g[c]):>20.3f}" for _, c in models))
    print(f"{'ALL':<6}{len(pred):>7}" +
          "".join(f"{safe_auc(pred.y, pred[c]):>20.3f}" for _, c in models))
 
    print("\n=== AUC by stock (all test years pooled) ===")
    print(f"{'Stock':<12}" + "".join(f"{n:>20}" for n, _ in models))
    for st, g in pred.groupby("stock"):
        print(f"{st:<12}" + "".join(f"{safe_auc(g.y, g[c]):>20.3f}" for _, c in models))
 
    base = pred.y.mean()
    print(f"\nShare of label = 1 in the test years: {100 * base:.1f}%  "
          f"(always predicting the more common class would be right "
          f"{100 * max(base, 1 - base):.1f}% of the time)")
 
    for name, col in models[:2]:
        print(f"\n=== Calibration: {name} (do predicted probabilities match reality?) ===")
        print(calibration_table(pred.y, pred[col]).round(3).to_string())
        hi = pred[pred[col] >= 0.55]
        if len(hi):
            print(f"Signals with probability >= 0.55: {len(hi)} days "
                  f"({100 * len(hi) / len(pred):.1f}% of days)   "
                  f"hit rate {100 * hi.y.mean():.1f}%   "
                  f"avg 5-day return {100 * hi.fwd_ret.mean():.2f}% "
                  f"(all days: {100 * pred.fwd_ret.mean():.2f}%, before costs)")
            if "excess_ret" in pred.columns:
                print(f"   Average 5-day return vs NIFTY on these signals: "
                      f"{100 * hi.excess_ret.mean():+.3f}%  "
                      f"(all days: {100 * pred.excess_ret.mean():+.3f}%, before costs)")
        else:
            print("No signals with probability >= 0.55.")
 
 
def main():
    RES_DIR.mkdir(exist_ok=True)
    rel = len(sys.argv) > 1 and sys.argv[1] == "rel"
    suffix = "_dataset_rel" if rel else "_dataset"
    out_name = "walk_forward_predictions_rel.csv" if rel else "walk_forward_predictions.csv"
    print("Label:", "stock beats NIFTY over 5 days (Experiment 2)" if rel
          else "stock rises more than 0.5% in 5 days (Experiment 1)")
    frames = []
    for name in STOCKS:
        path = PROC_DIR / f"{name}{suffix}.parquet"
        if not path.exists():
            print(f"{name}: {path} not found. Run labels.py (or labels_rel.py) first.")
            continue
        print(f"Training {name} ...", flush=True)
        pred = walk_forward(pd.read_parquet(path))
        if pred.empty:
            continue
        frames.append(pred.assign(stock=name))
    if not frames:
        return
    pred = pd.concat(frames)
    pred.to_csv(RES_DIR / out_name)
    print(f"\nTest period: {pred.index.min().date()} to {pred.index.max().date()} "
          f"(locked final period after {DEV_END} NOT used)")
    report(pred)
    print(f"\nPredictions saved to {RES_DIR / out_name}")
 
 
if __name__ == "__main__":
    main()