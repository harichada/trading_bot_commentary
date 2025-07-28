# CommentaryType Enum Reference

Valid CommentaryType values in trading_bot_commentary_updated.py:

```python
class CommentaryType(Enum):
    MARKET_ANALYSIS = "market_analysis"      # General market analysis
    SIGNAL_GENERATION = "signal_generation"  # Trading signal generation
    RISK_ASSESSMENT = "risk_assessment"      # Risk analysis
    DECISION = "decision"                    # Trading decisions
    TECHNICAL = "technical"                  # Technical analysis
    FUNDAMENTAL = "fundamental"              # Fundamental analysis
    PSYCHOLOGY = "psychology"                # Market psychology/sentiment
    WARNING = "warning"                      # Warnings and alerts
    OPPORTUNITY = "opportunity"              # Trading opportunities
    ANOMALY = "anomaly"                      # Anomalies detected
```

## Mapping Guide for Scalping ML:

- `MODEL_INIT` → `MARKET_ANALYSIS` - Model initialization
- `ML_SIGNAL` → `SIGNAL_GENERATION` - ML trading signals
- `ML_RESULT` → `DECISION` - Trade results/decisions (NOT TRADE_EXECUTION)
- `MODEL_TRAINING` → `MARKET_ANALYSIS` - Model training updates
- `DRIFT_DETECTION` → `WARNING` - Model drift warnings
- `RETRAIN_SCHEDULED` → `MARKET_ANALYSIS` - Retraining notifications
- `DATA_COLLECTION` → `MARKET_ANALYSIS` - Data collection updates
- `MODEL_VALIDATION` → `TECHNICAL` - Validation results
- `BACKTEST_COMPLETE` → `DECISION` - Backtesting results

Note: There is no `TRADE_EXECUTION` type. Use `DECISION` for trade-related commentary.