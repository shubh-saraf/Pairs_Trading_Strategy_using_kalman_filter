# Final Cointegration Project

This project focuses on identifying and backtesting mean-reverting currency pairs using Johansen Cointegration tests and Kalman Filters.

## Project Structure

- `pair_screener.py`: Screens for cointegrated pairs from a set of tickers.
- `recent_cointegration_test.py`: Validates the cointegration of screened pairs and ensures the hedge ratio is within acceptable bounds for a recent data window.
- `kalman_batch_backtest.py`: Runs a batch backtest on recently cointegrated pairs using a Kalman filter trading strategy.

## Workflow

1.  **Screening**: Run `pair_screener.py` to identify potential candidates.
2.  **Recent Validation**: Run `recent_cointegration_test.py` to filter for pairs that maintain cointegration in recent data. This generates `recently_cointegrated_pairs.csv`.
3.  **Backtesting**: Run `kalman_batch_backtest.py` to evaluate the performance of the trading strategy.

## Key Features

- Automated data download for Nifty 50 and Nifty Next 50.
- Multi-staged filtering process (Pearson correlation, Johansen test, OU half-life).
- Dynamic hedge ratio estimation via Kalman Filter.
- Comprehensive performance reporting.
