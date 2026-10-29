
import sys
from pathlib import Path
 
import pandas as pd
 
import pooled as P
from model import NON_FEATURES, PROC_DIR, RES_DIR
 
RAW_DIR = Path("data/raw")
KEEP = ["y", "fwd_ret", "excess_ret", "nifty_fwd_ret", "ret_5"]   # ret_5 kept only for the momentum baseline
 
 
def market_features(close):
    """NIFTY context known at the close of day t."""
    r = close.pct_change()
    f = pd.DataFrame(index=close.index)
    f["m_ret_1"] = r
    f["m_ret_5"] = close.pct_change(5)
    f["m_ret_20"] = close.pct_change(20)
    f["m_dist_sma50"] = close / close.rolling(50).mean() - 1
    f["m_vol20"] = r.rolling(20).std()
    return f
 
 
def add_rank_features(big, feat_cols):
    """Rank each feature across the stocks present on the same date (0 to 1)."""
    rk = big.groupby(level=0)[feat_cols].rank(pct=True)
    for c in feat_cols:
        big["r_" + c] = rk[c].values          # same row order as big
    return ["r_" + c for c in feat_cols]
 
 
def attach_market(big, nifty_close):
    mk = market_features(nifty_close)
    mk = mk.reindex(mk.index.union(big.index.unique())).ffill().reindex(big.index)
    for c in mk.columns:
        big[c] = mk[c].values
    return list(mk.columns)
 
 
def build_frames(dev, mode, nifty_close=None):
    big = pd.concat([d.assign(stock=n) for n, d in dev.items()])
    feat_cols = [c for c in big.columns if c not in NON_FEATURES + ["stock"]]
    use = add_rank_features(big, feat_cols)
    if mode == "rank_mkt":
        use += attach_market(big, nifty_close)
    big = big[KEEP + use + ["stock"]].dropna()
    frames = {n: g.drop(columns="stock") for n, g in big.groupby("stock")}
    return frames, use
 
 
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rank"
    if mode not in ("rank", "rank_mkt"):
        print("Use: python src/pooled_xs.py rank   or   python src/pooled_xs.py rank_mkt")
        return
    names = sorted(p.name.replace("_dataset_rel.parquet", "")
                   for p in PROC_DIR.glob("*_dataset_rel.parquet"))
    dev = P.load_dev(names)
    if not dev:
        print("No *_dataset_rel.parquet files found. Run build_universe.py first.")
        return
    nifty_close = None
    if mode == "rank_mkt":
        path = RAW_DIR / "NIFTY.parquet"
        if not path.exists():
            print("data/raw/NIFTY.parquet not found. Run python src/data_index.py first.")
            return
        nifty_close = pd.read_parquet(path).Close
 
    frames, use = build_frames(dev, mode, nifty_close)
    print(f"Experiment 4{'a' if mode == 'rank' else 'b'}: {len(frames)} stocks, {len(use)} features "
          f"({'ranks only' if mode == 'rank' else 'ranks + NIFTY market features'})")
    pred = P.walk_forward_pooled(frames, feature_cols=use)
    if pred.empty:
        return
    RES_DIR.mkdir(exist_ok=True)
    out = RES_DIR / f"walk_forward_predictions_xs_{mode}.csv"
    pred.to_csv(out)
    print(f"Test period: {pred.index.min().date()} to {pred.index.max().date()} "
          "(locked final period NOT used)")
    P.report(pred)
    print(f"\nPredictions saved to {out}")
 
 
if __name__ == "__main__":
    main()
 