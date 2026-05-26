"""Research subtree — backtest, feature engineering, and model
training infrastructure for the side classifier.

v-research-isolation-2026-05-13. Strict separation from the live bot:

  - This subtree imports from ``core.classifier`` but NEVER from
    ``core.engine`` or anything that touches the broker.
  - The live bot does not import ``research.*`` for any reason.
  - Backtest reports land in ``docs/training_runs/`` or
    ``backtest_results/classifier/``, never overwriting the strategy
    backtests at ``backtest_results/latest_report.json``.

Entry points:
  - ``research/classifier_backtest.py`` — replay decisions against
    labelled historical data
  - (future) ``research/train_tft.py`` — train the TFT model on
    yfinance + Alpaca news data once torch+CUDA is installed
"""
