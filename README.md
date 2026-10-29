Stock AI: can price data rank Indian large caps?

A research project that tests, with strict out-of-sample rules, whether daily price and volume data for large Indian stocks can predict which stocks will beat the NIFTY 50, and whether that survives trading costs.

Result in one line: no. Across 7 experiments there is no tradable edge, and the final diagnostics show that the small positive signals seen along the way cannot be told apart from chance. That is a useful, honest result.

What was built
Data: daily prices from Yahoo Finance (yfinance) for 45 large-cap NSE stocks plus NIFTY 50, stored as Parquet.
Features: 16 basic indicators (returns, moving-average distance, RSI, MACD, volatility, ATR, volume, gap), later 31 (long momentum, 52-week position, market breadth, NIFTY regime) with cross-stock ranks.
Labels: price up more than 0.5% after 5 days; later "beats NIFTY over 5 / 10 / 20 days".
Models: Random Forest, XGBoost, Logistic Regression, XGBoost ranker; one model per stock, then one pooled model.
Validation: walk-forward, test years 2018-2025, embargo equal to the label horizon, final period after 14 Oct 2025 locked and never opened.
Backtests: next-open entry, ATR stop and target, position sizing, 0.30% round-trip cost; and a top-10 monthly portfolio against an equal-weight benchmark.
Checks: a truncation test for look-ahead leakage, label tests, planted-signal controls (the pipeline detects a planted signal at AUC 0.7 to 0.9, and noise gives about 0.50).
Results
Experiment	Change	AUC
1	5 stocks, "up 0.5% in 5 days"	0.507 / 0.509
2	label: beats NIFTY	0.523 / 0.521
3	one pooled model, 5 then 45 stocks	0.506 / 0.503 and 0.509 / 0.511
4	cross-stock rank features, plus NIFTY features	0.510 / 0.508 and 0.509 / 0.508
5	10-day and 20-day horizons	0.505 / 0.503 and 0.495 / 0.497
6	31 features, Logistic, XGBoost, ranker, top-10 portfolio	0.510 / 0.509 / 0.507
7	confidence intervals on Experiment 6	see below

The bar set in advance (AUC above 0.53 in at least 5 of 8 years, and an economic advantage of at least 0.30% per holding period before costs) was never met.

Experiment 7 (95% block-bootstrap intervals, Version 2, 20-day label):

Model	Mean IC	Net excess vs universe per month	Information ratio
Logistic Regression	+0.017 [-0.012, +0.044]	-0.21% [-0.68%, +0.24%]	-0.31
XGBoost classifier	+0.022 [-0.007, +0.053]	-0.14% [-0.67%, +0.45%]	-0.20
XGBoost ranker	+0.014 [-0.010, +0.041]	-0.24% [-0.67%, +0.19%]	-0.36

Every interval includes zero. Gross excess before costs was between -0.05% and +0.04% per month, so there was nothing for costs to eat into. The top-10 portfolio also had deeper drops than the universe in 2022 and 2024.

Limitations
The universe is today's surviving large caps. Delisted stocks and past index membership are not available in free data, so historical results are probably flattered (the direction of the bias for each strategy is not certain).
Overlapping labels make neighbouring days look more informative than they are; non-overlapping checks and block bootstrap were used for that reason.
Daily data only, no news, no fundamentals, no order-book data. Only price-based inputs were tested.
Eight test years were reused across experiments, so any "good" result would also need the locked period.
Conclusion

Simple price-based models on large Indian stocks showed no reliable, cost-adjusted edge. Do not trade real money with these models. The locked final period stays unopened because no model passed the bar.

How to run (Windows, from the project folder with the venv active)
python src/data.py
python src/data_index.py
python src/build_universe.py
python src/pooled.py all
python src/v2.py 20
python src/diagnostics.py

Data and results folders are not committed (see .gitignore). Keep the repository private (Yahoo data terms). Not financial advice; this is a personal research project.