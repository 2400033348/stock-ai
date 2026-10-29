"""
Stock AI research dashboard (Streamlit). Reads the saved results; nothing is retrained.

Install once (venv active):   pip install streamlit
Run from the project folder:  streamlit run src/dashboard.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from diagnostics import block_bootstrap_mean, sharpe, yearly_drawdown
from model import safe_auc
from v2 import daily_ic, topk_backtest
import paper

H = 20
RES = Path("results")
RUNS = {"Experiment 6: Version 2": "walk_forward_predictions_v2_h20.csv",
        "Experiment 8: Version 2 + outside markets": "walk_forward_predictions_v2_h20_exp8.csv"}
MODELS = {"Logistic Regression": "s_logit", "XGBoost classifier": "s_xgb", "XGBoost ranker": "s_rank"}

EXPERIMENTS = pd.DataFrame([
    ("1", "5 stocks, 'up 0.5% in 5 days', per-stock models", "0.507 / 0.509", "No usable edge"),
    ("2", "Label: beats NIFTY over 5 days", "0.523 / 0.521", "Edge about 0.10% per 5 days, below the 0.30% cost"),
    ("3", "One pooled model, 5 then 45 stocks", "0.509 / 0.511 (45 stocks)", "0 of 8 years above 0.53"),
    ("4", "Cross-stock ranks, plus NIFTY features", "0.509 / 0.508", "No improvement"),
    ("5", "10-day and 20-day horizons", "0.505 / 0.503 and 0.495 / 0.497", "Worse"),
    ("6", "31 features, Logistic, XGBoost, ranker, top-10 portfolio", "0.510 / 0.509 / 0.507", "Net -0.14% to -0.24% a month"),
    ("7", "Confidence intervals on Experiment 6", "-", "All intervals include zero"),
    ("8", "Plus S&P 500, USD/INR, Brent, VIX, betas", "0.507 / 0.508 / 0.507", "Net -0.19% to -0.42% a month"),
], columns=["#", "What changed", "AUC (first two models)", "Outcome"])


@st.cache_data
def load(name):
    p = RES / name
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()


@st.cache_data
def portfolio(name, col):
    return topk_backtest(load(name), col, H)


@st.cache_data
def intervals(name, col):
    pred = load(name)
    ic = daily_ic(pred, col)
    bt = portfolio(name, col)
    ex = bt.net - bt.ew
    return ic, ex, block_bootstrap_mean(ic.values, H), block_bootstrap_mean(ex.values, 3)


st.set_page_config(page_title="Stock AI research", layout="wide")
st.title("Can price data rank Indian large caps?")
st.caption("A walk-forward research project: 45 NSE stocks, 2018 to Sep 2025 test period, locked final period unopened.")

tab_paper, tab_over, tab_models, tab_port, tab_unc = st.tabs(
    ["Paper trading", "Conclusion", "Model skill", "Portfolio", "Uncertainty"])

with tab_paper:
    st.subheader("Paper trading: record the models' picks, score them later, learn from the misses")
    st.caption("No real money and no orders. Picks are recorded in paper/ledger.csv; entry at the next day's open, exit 20 trading "
               "days later, 0.30% cost on replaced stocks, compared with all 45 stocks equal weight. Judge only after 24+ periods.")
    led = pd.read_csv(paper.LEDGER, parse_dates=["signal_date"]) if paper.LEDGER.exists() else None
    if led is None:
        st.info("No signal yet. Press the button to record the first one.")
    else:
        last = led.signal_date.max()
        days = int(np.busday_count(last.date(), pd.Timestamp.today().date()))
        st.write(f"Last signal: **{last.date()}** (about {days} trading days ago). Next one is due after 20 trading days.")
    b1, b2 = st.columns(2)
    if b1.button("Generate a new signal (downloads data, retrains, takes a few minutes)"):
        with st.spinner("Refreshing data and training the three models ..."):
            status, res = paper.run_signal()
        if status == "not_due":
            st.warning(f"Not due yet: {res} more trading days.")
        else:
            st.success(f"Signal recorded for {res[0].date()}. Buy at the next trading day's open.")
            led = pd.read_csv(paper.LEDGER, parse_dates=["signal_date"])
    if b2.button("Score finished periods (downloads data)"):
        with st.spinner("Scoring ..."):
            out = paper.run_score()
        if out is not None:
            st.success("Scored.")
        else:
            st.info("No holding period has finished yet.")

    if led is not None:
        st.write("Latest picks")
        latest = led[led.signal_date == led.signal_date.max()]
        cols = st.columns(len(paper.MODELS))
        for (mname, mcol), c in zip(paper.MODELS, cols):
            c.write(mname)
            keep = [k for k in ["rank", "stock", "score", "vol_rank", "mom_rank"] if k in latest.columns]
            c.dataframe(latest[latest.model == mcol][keep].round(2), hide_index=True, width="stretch")
        st.caption("vol_rank / mom_rank: how volatile / how strong in recent momentum the stock was among all 45 (1.00 = highest).")

    if paper.SCORE.exists():
        sc = pd.read_csv(paper.SCORE, parse_dates=["signal_date"]).set_index("signal_date")
        n = len(sc)
        st.write(f"Finished periods: **{n}** of 24 needed before any judgement")
        st.progress(min(n / 24, 1.0))
        rows = []
        for mname, mcol in paper.MODELS:
            x = sc[mcol]
            se = x.std() / np.sqrt(n) if n > 1 else np.nan
            rows.append({"Model": mname, "Net excess per period": f"{100 * x.mean():+.2f}%",
                         "Standard error": f"{100 * se:.2f}%", "Beat universe": f"{100 * (x > 0).mean():.0f}% of periods"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write("Cumulative net excess over the universe (%)")
        st.line_chart(100 * sc[[m for _, m in paper.MODELS]].rename(columns={m: nm for nm, m in paper.MODELS}).cumsum())
        if n < 24:
            st.caption("Too few periods: a good or bad run now is mostly luck.")

    dpath = paper.PAPER / "detail.csv"
    if dpath.exists():
        d = pd.read_csv(dpath)
        names = {c: nm for nm, c in paper.MODELS}
        st.subheader("Where did it go wrong?")
        by = d.groupby("model").agg(picks=("excess", "size"), hit_rate=("excess", lambda x: 100 * (x > 0).mean()),
                                    avg_excess=("excess", lambda x: 100 * x.mean()))
        by.index = [names.get(i, i) for i in by.index]
        st.write("Share of picks that beat the universe (a coin flip is about 50%)")
        st.dataframe(by.round(2).rename(columns={"hit_rate": "Hit rate %", "avg_excess": "Average excess %", "picks": "Picks"}),
                     width="stretch")
        wcols = [k for k in ["signal_date", "model", "stock", "rank", "ret", "excess", "vol_rank", "mom_rank"] if k in d.columns]
        worst = d.sort_values("excess").head(10)[wcols]
        worst = worst.assign(ret=(100 * worst.ret).round(1), excess=(100 * worst.excess).round(1))
        st.write("The 10 worst picks (return and excess in %)")
        st.dataframe(worst, hide_index=True, width="stretch")
        tc = [k for k in ["excess", "vol_rank", "mom_rank", "rank"] if k in d.columns]
        corr = d[tc].corr()["excess"].drop("excess").round(3)
        st.write("Do misses share a pattern? Correlation of the pick's excess return with its traits")
        st.dataframe(corr.rename("correlation").to_frame().T, width="stretch")
        st.caption("A clearly negative number for vol_rank would mean the volatile picks are the ones losing. With few picks these "
                   "numbers are noisy: look for patterns that repeat over many periods before acting on them.")
        bad = d.groupby("stock").excess.agg(["count", "mean"]).query("count >= 3").sort_values("mean").head(5)
        if len(bad):
            st.write("Stocks picked repeatedly that kept lagging (average excess per pick)")
            st.dataframe((100 * bad["mean"]).round(2).rename("average excess %").to_frame(), width="stretch")

with tab_over:
    st.subheader("No tradable edge was found")
    st.write("Across eight experiments, predictions stayed close to a coin flip (AUC about 0.51), the top-10 portfolios "
             "did not beat simply holding all 45 stocks after costs, and every confidence interval includes zero. "
             "The project is a rigorous negative result, not a trading system.")
    st.dataframe(EXPERIMENTS, hide_index=True, width="stretch")
    st.markdown("**Bar set in advance:** AUC above 0.53 in at least 5 of 8 years, a top-10 net excess return of at least "
                "+0.30% a month in at least 5 of 8 years, and a 95% interval that excludes zero. It was never met.")
    st.markdown("**Limitations:** the universe is today's surviving large caps (survivorship bias); returns are measured "
                "from the signal day's close; daily price data only; FII/DII flows not tested.")

run_name = st.sidebar.selectbox("Experiment", list(RUNS))
pred = load(RUNS[run_name])
if pred is None:
    for t in (tab_models, tab_port, tab_unc):
        with t:
            st.info(f"results/{RUNS[run_name]} not found. Run the experiment first.")
    st.stop()
model_name = st.sidebar.selectbox("Model", list(MODELS))
col = MODELS[model_name]

with tab_models:
    st.subheader(f"{model_name}: how well does it rank stocks?")
    yearly = pred.groupby(pred.index.year).apply(lambda g: safe_auc(g.y, g[col]))
    ic, ex, ic_ci, ex_ci = intervals(RUNS[run_name], col)
    ic_year = ic.groupby(ic.index.year).mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall AUC (0.50 = coin flip)", f"{safe_auc(pred.y, pred[col]):.3f}")
    c2.metric("Mean daily IC", f"{ic.mean():+.4f}")
    c3.metric("Years with AUC above 0.53", f"{int((yearly > 0.53).sum())} of {len(yearly)}")
    left, right = st.columns(2)
    left.write("AUC by test year")
    left.bar_chart(yearly.rename("AUC"))
    right.write("Mean information coefficient by year")
    right.bar_chart(ic_year.rename("IC"))
    st.caption("The signal flips sign from year to year, which is what noise looks like.")

with tab_port:
    st.subheader("Top-10 portfolio, rebalanced every 20 trading days, 0.30% cost per replaced stock")
    bt = portfolio(RUNS[run_name], col)
    curves = pd.DataFrame({"Top-10 portfolio (net)": (1 + bt.net).cumprod(),
                           "Equal-weight universe": (1 + bt.ew).cumprod(),
                           "NIFTY": (1 + bt.nifty).cumprod()})
    st.write("Growth of 1 rupee")
    st.line_chart(curves)
    dd = (curves / curves.cummax() - 1) * 100
    st.write("Drawdown (%)")
    st.line_chart(dd)
    ppy = 252 / H
    summary = pd.DataFrame({
        "Sharpe": [sharpe(bt.net, ppy), sharpe(bt.ew, ppy), sharpe(bt.nifty, ppy)],
        "Average per month (%)": [100 * bt.net.mean(), 100 * bt.ew.mean(), 100 * bt.nifty.mean()]},
        index=["Top-10 portfolio (net)", "Equal-weight universe", "NIFTY"]).round(2)
    st.dataframe(summary, width="stretch")
    st.write("Net excess over the universe by year (% per month)")
    st.bar_chart((100 * (bt.net - bt.ew)).groupby(bt.index.year).mean().rename("excess"))
    st.caption(f"Average turnover {100 * bt.turnover.mean():.0f}% of the portfolio per rebalance. "
               "The universe is today's survivors, so its returns are flattering.")

with tab_unc:
    st.subheader("Is any of this distinguishable from luck?")
    rows = []
    for m, c in MODELS.items():
        ic, ex, ic_ci, ex_ci = intervals(RUNS[run_name], c)
        rows.append({"Model": m,
                     "Mean IC": f"{ic.mean():+.4f}", "IC 95% interval": f"[{ic_ci[0]:+.4f}, {ic_ci[1]:+.4f}]",
                     "Net excess / month": f"{100 * ex.mean():+.3f}%",
                     "95% interval": f"[{100 * ex_ci[0]:+.3f}%, {100 * ex_ci[1]:+.3f}%]",
                     "Excludes zero?": "yes" if (ex_ci[0] > 0 or ex_ci[1] < 0) else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.write("Worst drop inside each year, selected model (portfolio, %)")
    st.dataframe(pd.Series(yearly_drawdown(portfolio(RUNS[run_name], col).net)).round(1).rename("Worst drop").to_frame().T,
                 width="stretch")
    st.caption("Intervals use a block bootstrap, so overlapping returns do not make the result look more certain than it is.")