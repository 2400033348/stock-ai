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