#!/usr/bin/env python3
"""
Trading Bot with Real-Time Commentary System
Explains its thinking process in detail while analyzing markets

This file is the thin orchestrator that imports from the modular packages.
All business logic has been extracted into:
  - core/       : models, config, commentary, brain, websocket, engine
  - analysis/   : anomaly detection, behavioral analysis, technical analysis, screener
  - ml/         : feature engineering, ML models, predictor
  - strategies/ : trading strategies (breakout, momentum, mean reversion, news)
  - risk/       : risk management, backtesting, performance analysis
  - data_providers/ : Schwab, yfinance, news aggregation
  - api/        : FastAPI routes and endpoints
  - templates/  : HTML dashboard templates
"""

# ---------------------------------------------------------------------------
# Re-export everything that external modules expect from this file.
# This ensures backward compatibility with existing imports like:
#   from trading_bot_commentary_updated import TradingEngineWithCommentary
# ---------------------------------------------------------------------------

# Core models and enums
from core.models import (
    TradingMode,
    CommentaryType,
    SignalType,
    NewsImpact,
    TradingSignal,
    Position,
    MarketData,
    NewsItem,
)

# Configuration
from core.config import (
    Config,
    ConfigManager,
    config_manager,
    config,
    setup_logging,
    logger,
    CircuitBreaker,
    ErrorRecovery,
    error_handler,
)

# Commentary system
from core.commentary import TradingCommentary, CommentarySystem

# Brain / learning system
from core.brain import TradingMemory, TradingBrain

# WebSocket connection manager
from core.websocket_manager import ConnectionManager

# The main trading engine
from core.engine import TradingEngineWithCommentary

# Analysis components
from analysis.anomaly import AnomalyDetector, DataValidator
from analysis.behavioral import BehavioralAnalyzer
from analysis.exit_managers import AdvancedExitManager, DynamicExitManager
from analysis.alternative_data import AlternativeDataIntegrator, MarketNeutralStrategies
from analysis.technical import TechnicalAnalyzerWithCommentary
from analysis.screener import StockScreener

# ML components
from ml.features import (
    MarketRegimeDetector,
    ConceptDriftDetector,
    AdvancedFeatureEngineer,
    MLFeatureExtractor,
    MLTrainingRecord,
    MLPrediction,
)
from ml.models import HybridTradingModel, IntegratedMLModel
from ml.predictor import MLPredictorWithCommentary

# Risk management
from risk.manager import RiskManagerWithCommentary
from risk.backtest import BacktestResult, BacktestEngine, PerformanceAnalyzer

# Strategies
from strategies.base import TradingStrategyWithCommentary
from strategies.builtin import (
    NewsSignalStrategy,
    BreakoutStrategyWithCommentary,
    MeanReversionStrategyWithCommentary,
    MomentumStrategyWithCommentary,
)
from strategies.news_strategy import FreeNewsAggregator, FreeNewsSignalStrategy

# Data providers
from data_providers.schwab import SchwabDataProvider
from data_providers.realtime import RealTimeDataProvider, DummyDataProvider

# External module re-exports
from trading_exceptions import *
from error_recovery import ErrorRecoveryManager

# FastAPI app and routes
from api.routes import app, main, connection_manager as _ws_connection_manager

# Re-export the module-level connection_manager used by other files
connection_manager = _ws_connection_manager

# Load dashboard HTML for backward compatibility
try:
    import os as _os
    _template_path = _os.path.join(_os.path.dirname(__file__), 'templates', 'dashboard.html')
    with open(_template_path, 'r') as _f:
        DASHBOARD_HTML_WITH_COMMENTARY = _f.read()
except FileNotFoundError:
    DASHBOARD_HTML_WITH_COMMENTARY = "<html><body><h1>Dashboard not found</h1></body></html>"


if __name__ == "__main__":
    main()
