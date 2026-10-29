# Experiments log

## Experiment 1 (17 Oct 2026): v1 baseline
- Data: 5 stocks (RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK), daily, 2012 to 14 Oct 2025
- Label: price up more than 0.5% after 5 trading days (47.7% of test days)
- Features: 16 (returns, SMA distance, RSI, MACD, volatility, ATR, volume, gap)
- Models: Random Forest, XGBoost, one per stock, walk-forward 2018-2025
- Result: AUC 0.507 (RF), 0.509 (XGB), 0.498 (momentum baseline)
- Calibration: poor. Predicted 0.36 to 0.66 all came true about 47-50% of the time
- Signals at 0.55 or above: hit rate 48.2% vs 47.7% base, average return below buy-every-day
- Conclusion: no usable edge. Go/no-go rule NOT met
- Locked final period: not used
## Experiment 1b (19 Oct 2026): backtest of v1
- Rules: prob >= 0.55, entry next open, stop 1.5 ATR, target 3 ATR, 5-day exit, 0.30% cost
- Avg net per trade: RF -0.13%, XGB +0.03%, always-long -0.07% (all within noise)
- At 0.40% cost: RF -0.23%, XGB -0.07%, always-long -0.17%
- XGB positive 2018-2021, negative 2022-2025; RF positive only in 2023
- Exits: ~64% time, ~25% stop, ~6% target for every strategy
- Conclusion: no-go. Locked period not opened
## Experiment 2 (19 Oct 2026): relative label
- Change: label = 1 if the stock's 5-day return beats NIFTY 50's 5-day return. Features, models and walk-forward unchanged
- Why: removes the market-wide move that the features can't predict
- Success: AUC above 0.53 in most test years, and signals with a positive average return vs NIFTY
- Locked period: not used
## Experiment 2 result (19 Oct 2026)
- AUC: RF 0.523, XGB 0.521 (Exp 1: 0.507, 0.509). RF >= 0.50 in 8 of 8 years
- AUC above 0.53 in only 3 of 8 years for each model: AUC bar NOT met
- Signals (prob >= 0.55): +0.21% to +0.24% vs NIFTY per 5 days, all days +0.12%
- Edge over average day is about 0.10% per 5 days, versus 0.30% cost: not tradable
- Caution: selection bias (5 current large caps), small AUC gain within noise
- Locked period: not used
## Experiment 3 (19 Oct 2026): pooled model, more stocks
- Change: ONE model trained on many stocks together (3a: original 5; 3b: about 40 more large caps)
- Same features, same relative label (beat NIFTY over 5 days), same walk-forward 2018-2025
- Bar set in advance: AUC above 0.53 in at least 5 of 8 years, AND signals beating the all-days
  average by at least +0.30% per 5 days before costs
- Caution: the 3b list is today's large caps, so survivorship bias will flatter results
- Locked period: not used
## Experiment 3 result (19 Oct 2026)
- 3a pooled model, 5 stocks: AUC RF 0.506, XGB 0.503. Worse than per-stock 0.523
- 3b pooled model, 45 stocks: AUC RF 0.509, XGB 0.511, 0 of 8 years above 0.53, none below 0.50
- Signals advantage: top fifth vs all +0.05% (RF) and +0.08% (XGB) per 5 days; RF signals >= 0.55 +0.21%
- Bar (AUC >0.53 in 5 of 8 years and >= +0.30% per 5 days): NOT met
- Likely a very small real signal (about 0.51), far below the 0.30% round-trip cost
- Survivorship bias in the 45-stock list flatters the result
- Locked period: not used
## Experiment 4 result (27 Oct 2026)
- 4a rank features: AUC RF 0.510, XGB 0.508, 0 of 8 years above 0.53
- 4b ranks + NIFTY market features: AUC RF 0.509, XGB 0.508, 0 of 8 years above 0.53
- Top fifth vs all: +0.06% to +0.08% per 5 days (positive in 7 of 8 years in 4a), cost is 0.30%
- Bar NOT met. Market features added nothing
- Conclusion: the 16 features carry about 0.51 AUC regardless of representation
- Locked period: not used
## Experiment 5 result (27 Oct 2026)
- 10-day label: AUC RF 0.505, XGB 0.503, 0 of 8 years above 0.53, advantage -0.09% / -0.16% per 10 days
- 20-day label: AUC RF 0.495, XGB 0.497, advantage -0.20% / -0.24% per 20 days, non-overlapping AUC below 0.50
- Bar NOT met. The signal fades with horizon instead of growing
- Overall (Experiments 1-5): 8 set-ups, AUC 0.495 to 0.523, no consistent edge, best effect under 0.1% per 5 days vs 0.30% cost
- DECISION: stop model experiments. Nothing passed, so no model is frozen and the locked period stays unopened
- Locked period: not used
## Experiment 6: Version 2 (31 features incl. long momentum, breadth, NIFTY regime; ranks; Logistic, XGBoost, XGBoost ranker; 20-day relative label; top-10 monthly portfolio, 0.30% cost per replaced stock)
Result: AUC 0.510 / 0.509 / 0.507; years above 0.53: 2 / 2 / 1 of 8. Mean IC +0.017 / +0.022 / +0.014.
Top-10 net excess over the equal-weight universe: -0.21% / -0.14% / -0.24% per month.
Verdict: bar NOT met. No evidence of a tradable edge. Locked period still unopened.
## Experiment 8: Version 2 plus outside-market information (29 Oct 2026, src/exp8.py)
- Added S&P 500, USD/INR, Brent crude and US VIX changes (lagged one step so they were known before India's close) and 60-day betas to NIFTY, USD/INR and crude. 42 features in total.
- AUC 0.507 / 0.508 / 0.507; years above 0.53: 2 / 2 / 0 of 8. Mean IC +0.011 / +0.018 / +0.014.
- Net excess vs equal-weight universe: -0.26% / -0.19% / -0.42% per month; information ratio -0.40 / -0.28 / -0.68.
- All 95% block-bootstrap intervals include zero. Gross excess -0.11% / -0.02% / -0.23% per month.
- Verdict: bar NOT met; slightly worse than Experiment 6. Locked period: not used.
- Final conclusion: no tradable edge found from price, volume, market regime or outside-market data (FII/DII flows not tested, no free source).
## Experiment 9: US small caps (29 Oct 2026, src/exp9.py)
- S&P SmallCap 600 (100 most liquid with history before 2014), benchmark IWM, 20-day label, cost 0.40%.
- AUC 0.495 / 0.492 / 0.502; years above 0.53: 1 / 0 / 0 of 8. Mean IC +0.007 / +0.005 / +0.009 (intervals include zero).
- Net excess vs universe: -0.87% / +0.22% / +1.04% per month. Ranker positive in 6 of 8 years but interval [-0.34%, +2.62%] includes zero.
- Verdict: bar NOT met (AUC and interval parts failed). Result driven mainly by 2020; best of three models chosen after the fact; survivorship bias severe. Locked period: not used.
## Experiment 9b: baseline check (29 Oct 2026, src/check_exp9.py)
- Ranker +1.04% per month vs "10 most volatile stocks" +2.48% (positive in 8 of 8 years) vs random 95th percentile +0.29%.
- Ranker picks had average volatility rank 0.71; without 2020 the ranker falls to +0.43%.
- Verdict: ranker adds no value beyond a simple rule. The volatility result contradicts the literature and is most likely an artifact of survivorship bias and of selecting stocks by recent dollar volume (look-ahead). Experiment 9 is not trusted. Locked period: not used.
- Lesson: a good-looking backtest in a survivor-only universe is a warning sign, not a finding.