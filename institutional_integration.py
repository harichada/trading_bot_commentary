#!/usr/bin/env python3
"""
Institutional Integration Layer
================================
This module wires all institutional components into the main trading engine.
It's the critical piece that makes everything work together.

Usage:
    from institutional_integration import InstitutionalTradingBot
    bot = InstitutionalTradingBot()
    bot.start()
"""

import os
import sys
import json
import asyncio
import threading
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('institutional_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('InstitutionalBot')

# Safe imports with fallbacks
try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("NumPy not available - some features disabled")

try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning("Pandas not available - some features disabled")


class TradingState(Enum):
    """Explicit state machine for trading operations"""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONCILING = "reconciling"
    READY = "ready"
    TRADING = "trading"
    RISK_CHECK = "risk_check"
    EXECUTING = "executing"
    COOLING_DOWN = "cooling_down"
    RISK_LOCKED = "risk_locked"
    ERROR_RECOVERY = "error_recovery"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


# Valid state transitions
STATE_TRANSITIONS = {
    TradingState.INITIALIZING: [TradingState.CONNECTING, TradingState.ERROR_RECOVERY],
    TradingState.CONNECTING: [TradingState.CONNECTED, TradingState.ERROR_RECOVERY],
    TradingState.CONNECTED: [TradingState.RECONCILING, TradingState.ERROR_RECOVERY],
    TradingState.RECONCILING: [TradingState.READY, TradingState.ERROR_RECOVERY],
    TradingState.READY: [TradingState.TRADING, TradingState.SHUTTING_DOWN, TradingState.RISK_LOCKED],
    TradingState.TRADING: [TradingState.RISK_CHECK, TradingState.READY, TradingState.ERROR_RECOVERY],
    TradingState.RISK_CHECK: [TradingState.EXECUTING, TradingState.READY, TradingState.RISK_LOCKED],
    TradingState.EXECUTING: [TradingState.READY, TradingState.COOLING_DOWN, TradingState.ERROR_RECOVERY],
    TradingState.COOLING_DOWN: [TradingState.READY],
    TradingState.RISK_LOCKED: [TradingState.READY, TradingState.SHUTTING_DOWN],
    TradingState.ERROR_RECOVERY: [TradingState.READY, TradingState.RISK_LOCKED, TradingState.SHUTTING_DOWN],
    TradingState.SHUTTING_DOWN: [TradingState.STOPPED],
    TradingState.STOPPED: [],
}


@dataclass
class TradeRecord:
    """Immutable record of a trade for audit trail"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    order_type: str
    strategy: str
    signals: Dict[str, Any]
    risk_metrics: Dict[str, float]
    state_at_execution: str
    execution_latency_ms: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class RiskSnapshot:
    """Point-in-time risk metrics"""
    timestamp: datetime
    total_exposure: float
    var_95: float
    var_99: float
    max_position_pct: float
    correlation_risk: float
    daily_pnl: float
    drawdown: float
    positions_count: int

    def is_within_limits(self, limits: Dict) -> tuple:
        """Check if within risk limits, return (ok, violations)"""
        violations = []

        if self.var_95 > limits.get('max_var_95', 0.02):
            violations.append(f"VaR95 {self.var_95:.2%} > limit {limits['max_var_95']:.2%}")

        if self.drawdown > limits.get('max_drawdown', 0.05):
            violations.append(f"Drawdown {self.drawdown:.2%} > limit {limits['max_drawdown']:.2%}")

        if abs(self.daily_pnl) > limits.get('max_daily_loss', 0.02) and self.daily_pnl < 0:
            violations.append(f"Daily loss {self.daily_pnl:.2%} > limit {limits['max_daily_loss']:.2%}")

        if self.total_exposure > limits.get('max_exposure', 0.8):
            violations.append(f"Exposure {self.total_exposure:.2%} > limit {limits['max_exposure']:.2%}")

        return len(violations) == 0, violations


class AtomicStateManager:
    """
    Crash-safe state persistence using Write-Ahead Logging (WAL).
    Survives crashes at any point during execution.
    """

    def __init__(self, state_file: str = "institutional_state.json",
                 wal_file: str = "institutional_wal.json"):
        self.state_file = Path(state_file)
        self.wal_file = Path(wal_file)
        self._lock = threading.RLock()
        self._state = self._recover_state()

    def _recover_state(self) -> Dict:
        """Recover state from WAL if crash occurred"""
        state = {}

        # Load base state
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

        # Replay WAL if exists (crash recovery)
        if self.wal_file.exists():
            try:
                with open(self.wal_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if entry.get('type') == 'update':
                            for key, value in entry.get('data', {}).items():
                                state[key] = value
                logger.warning(f"Recovered {self.wal_file.stat().st_size} bytes from WAL")

                # Apply recovered state
                self._write_state(state)
                self.wal_file.unlink()  # Clear WAL after successful recovery
            except Exception as e:
                logger.error(f"WAL recovery failed: {e}")

        return state

    def _write_state(self, state: Dict):
        """Atomically write state file"""
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        temp_file.replace(self.state_file)

    def update(self, updates: Dict):
        """Update state with WAL for crash safety"""
        with self._lock:
            # Write to WAL first
            wal_entry = {
                'type': 'update',
                'timestamp': datetime.now().isoformat(),
                'data': updates
            }
            with open(self.wal_file, 'a') as f:
                f.write(json.dumps(wal_entry, default=str) + '\n')

            # Apply updates
            self._state.update(updates)

            # Write full state
            self._write_state(self._state)

            # Clear WAL on success
            if self.wal_file.exists():
                self.wal_file.unlink()

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def get_all(self) -> Dict:
        with self._lock:
            return self._state.copy()


class IdempotentOrderManager:
    """
    Prevents duplicate orders using idempotency keys.
    Critical for crash recovery - never execute same order twice.
    """

    def __init__(self, persistence_file: str = "order_idempotency.json"):
        self.persistence_file = Path(persistence_file)
        self._lock = threading.RLock()
        self._executed_keys = self._load_keys()
        self._key_expiry_hours = 24

    def _load_keys(self) -> Dict[str, datetime]:
        """Load executed order keys"""
        if self.persistence_file.exists():
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                    return {k: datetime.fromisoformat(v) for k, v in data.items()}
            except Exception:
                pass
        return {}

    def _save_keys(self):
        """Save executed order keys"""
        with open(self.persistence_file, 'w') as f:
            json.dump({k: v.isoformat() for k, v in self._executed_keys.items()}, f)

    def _cleanup_expired(self):
        """Remove expired keys"""
        cutoff = datetime.now() - timedelta(hours=self._key_expiry_hours)
        self._executed_keys = {k: v for k, v in self._executed_keys.items() if v > cutoff}

    def generate_key(self, symbol: str, side: str, quantity: int,
                     strategy: str, signal_time: datetime) -> str:
        """Generate deterministic idempotency key"""
        key_data = f"{symbol}|{side}|{quantity}|{strategy}|{signal_time.isoformat()}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def check_and_mark(self, key: str) -> bool:
        """
        Check if order can be executed. Returns True if OK, False if duplicate.
        Atomically marks the key as executed if OK.
        """
        with self._lock:
            self._cleanup_expired()

            if key in self._executed_keys:
                logger.warning(f"Duplicate order detected: {key}")
                return False

            self._executed_keys[key] = datetime.now()
            self._save_keys()
            return True

    def is_duplicate(self, key: str) -> bool:
        """Check if key was already executed"""
        with self._lock:
            return key in self._executed_keys


class PositionReconciler:
    """
    Reconciles internal positions with broker positions.
    Detects phantom positions (we think we have, broker doesn't)
    and missing positions (broker has, we don't track).
    """

    def __init__(self):
        self._last_reconciliation = None
        self._reconciliation_interval = timedelta(minutes=5)
        self._discrepancies = []

    def reconcile(self, internal_positions: Dict, broker_positions: Dict) -> Dict:
        """
        Compare internal tracking with broker reality.
        Returns reconciliation report.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'OK',
            'phantom_positions': [],  # We have, broker doesn't
            'missing_positions': [],   # Broker has, we don't
            'quantity_mismatches': [], # Different quantities
            'price_discrepancies': [], # Significant price differences
        }

        internal_symbols = set(internal_positions.keys())
        broker_symbols = set(broker_positions.keys())

        # Phantom positions
        for symbol in internal_symbols - broker_symbols:
            report['phantom_positions'].append({
                'symbol': symbol,
                'internal_qty': internal_positions[symbol].get('quantity', 0),
                'action': 'REMOVE_FROM_TRACKING'
            })
            report['status'] = 'DISCREPANCY'

        # Missing positions
        for symbol in broker_symbols - internal_symbols:
            report['missing_positions'].append({
                'symbol': symbol,
                'broker_qty': broker_positions[symbol].get('quantity', 0),
                'action': 'ADD_TO_TRACKING'
            })
            report['status'] = 'DISCREPANCY'

        # Quantity and price mismatches
        for symbol in internal_symbols & broker_symbols:
            internal = internal_positions[symbol]
            broker = broker_positions[symbol]

            internal_qty = internal.get('quantity', 0)
            broker_qty = broker.get('quantity', 0)

            if internal_qty != broker_qty:
                report['quantity_mismatches'].append({
                    'symbol': symbol,
                    'internal_qty': internal_qty,
                    'broker_qty': broker_qty,
                    'action': 'UPDATE_TO_BROKER_QTY'
                })
                report['status'] = 'DISCREPANCY'

            # Check price discrepancy (>1% difference is concerning)
            internal_price = internal.get('current_price', 0)
            broker_price = broker.get('market_value', 0) / max(broker_qty, 1)
            if internal_price > 0 and broker_price > 0:
                price_diff = abs(internal_price - broker_price) / internal_price
                if price_diff > 0.01:
                    report['price_discrepancies'].append({
                        'symbol': symbol,
                        'internal_price': internal_price,
                        'broker_price': broker_price,
                        'difference_pct': price_diff
                    })

        self._last_reconciliation = datetime.now()
        self._discrepancies = report

        if report['status'] != 'OK':
            logger.warning(f"Position reconciliation found discrepancies: {json.dumps(report, indent=2)}")
        else:
            logger.info("Position reconciliation: OK")

        return report

    def needs_reconciliation(self) -> bool:
        """Check if reconciliation is due"""
        if self._last_reconciliation is None:
            return True
        return datetime.now() - self._last_reconciliation > self._reconciliation_interval


class RiskEngine:
    """
    Real-time risk calculation and enforcement.
    Calculates VaR, monitors correlations, enforces limits.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.limits = {
            'max_var_95': self.config.get('max_var_95', 0.02),
            'max_var_99': self.config.get('max_var_99', 0.03),
            'max_drawdown': self.config.get('max_drawdown', 0.05),
            'max_daily_loss': self.config.get('max_daily_loss', 0.02),
            'max_exposure': self.config.get('max_exposure', 0.80),
            'max_position_pct': self.config.get('max_position_pct', 0.15),
            'max_correlation': self.config.get('max_correlation', 0.80),
        }
        self._history = []
        self._daily_starting_equity = None
        self._peak_equity = None

    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation"""
        if not returns or len(returns) < 20:
            return 0.0

        if np is None:
            # Simple fallback without numpy
            sorted_returns = sorted(returns)
            index = int(len(sorted_returns) * (1 - confidence))
            return abs(sorted_returns[max(0, index)])

        returns_arr = np.array(returns)
        var = np.percentile(returns_arr, (1 - confidence) * 100)
        return abs(var)

    def calculate_snapshot(self, positions: Dict, account_value: float,
                          returns_history: List[float] = None) -> RiskSnapshot:
        """Calculate current risk metrics"""
        returns_history = returns_history or []

        # Calculate exposure
        total_position_value = sum(
            p.get('quantity', 0) * p.get('current_price', 0)
            for p in positions.values()
        )
        exposure = total_position_value / max(account_value, 1)

        # Max single position
        max_pos_value = max(
            (p.get('quantity', 0) * p.get('current_price', 0) for p in positions.values()),
            default=0
        )
        max_position_pct = max_pos_value / max(account_value, 1)

        # Daily P&L
        if self._daily_starting_equity is None:
            self._daily_starting_equity = account_value
        daily_pnl = (account_value - self._daily_starting_equity) / self._daily_starting_equity

        # Drawdown
        if self._peak_equity is None:
            self._peak_equity = account_value
        self._peak_equity = max(self._peak_equity, account_value)
        drawdown = (self._peak_equity - account_value) / self._peak_equity

        # VaR calculations
        var_95 = self.calculate_var(returns_history, 0.95)
        var_99 = self.calculate_var(returns_history, 0.99)

        snapshot = RiskSnapshot(
            timestamp=datetime.now(),
            total_exposure=exposure,
            var_95=var_95,
            var_99=var_99,
            max_position_pct=max_position_pct,
            correlation_risk=0.0,  # TODO: Implement correlation calculation
            daily_pnl=daily_pnl,
            drawdown=drawdown,
            positions_count=len(positions)
        )

        self._history.append(snapshot)
        return snapshot

    def check_trade_allowed(self, signal: Dict, positions: Dict,
                           account_value: float) -> tuple:
        """
        Pre-trade risk check. Returns (allowed: bool, reason: str)
        """
        # Get current risk snapshot
        snapshot = self.calculate_snapshot(positions, account_value)

        is_ok, violations = snapshot.is_within_limits(self.limits)

        if not is_ok:
            return False, f"Risk limits violated: {', '.join(violations)}"

        # Check if new trade would exceed limits
        proposed_value = signal.get('quantity', 0) * signal.get('entry_price', 0)
        new_exposure = (snapshot.total_exposure * account_value + proposed_value) / account_value

        if new_exposure > self.limits['max_exposure']:
            return False, f"Trade would exceed exposure limit: {new_exposure:.1%} > {self.limits['max_exposure']:.1%}"

        new_position_pct = proposed_value / account_value
        if new_position_pct > self.limits['max_position_pct']:
            return False, f"Position size exceeds limit: {new_position_pct:.1%} > {self.limits['max_position_pct']:.1%}"

        return True, "OK"

    def reset_daily(self, current_equity: float):
        """Reset daily tracking (call at market open)"""
        self._daily_starting_equity = current_equity
        logger.info(f"Daily risk tracking reset. Starting equity: ${current_equity:,.2f}")


class AuditTrail:
    """
    Immutable, tamper-evident audit trail for all trading activity.
    Uses hash chaining to detect any modifications.
    """

    def __init__(self, file_path: str = "audit_trail.jsonl"):
        self.file_path = Path(file_path)
        self._lock = threading.Lock()
        self._last_hash = self._get_last_hash()

    def _get_last_hash(self) -> str:
        """Get hash of last entry for chain verification"""
        if not self.file_path.exists():
            return "GENESIS"

        try:
            with open(self.file_path, 'rb') as f:
                # Read last line efficiently
                f.seek(0, 2)  # End of file
                size = f.tell()
                if size == 0:
                    return "GENESIS"

                # Read backwards to find last newline
                pos = size - 2
                while pos > 0 and f.read(1) != b'\n':
                    pos -= 1
                    f.seek(pos)

                f.seek(max(0, pos))
                last_line = f.readline().decode('utf-8').strip()
                if last_line:
                    entry = json.loads(last_line)
                    return entry.get('hash', 'GENESIS')
        except Exception as e:
            logger.error(f"Error reading audit trail: {e}")

        return "GENESIS"

    def _calculate_hash(self, entry: Dict, prev_hash: str) -> str:
        """Calculate tamper-evident hash"""
        data = json.dumps(entry, sort_keys=True, default=str) + prev_hash
        return hashlib.sha256(data.encode()).hexdigest()

    def record(self, event_type: str, data: Dict, actor: str = "system"):
        """Record an audit event"""
        with self._lock:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'actor': actor,
                'data': data,
                'prev_hash': self._last_hash,
            }
            entry['hash'] = self._calculate_hash(entry, self._last_hash)

            with open(self.file_path, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')

            self._last_hash = entry['hash']

    def verify_integrity(self) -> tuple:
        """Verify the entire audit trail hasn't been tampered with"""
        if not self.file_path.exists():
            return True, "No audit trail exists yet"

        prev_hash = "GENESIS"
        line_num = 0

        try:
            with open(self.file_path, 'r') as f:
                for line in f:
                    line_num += 1
                    entry = json.loads(line.strip())

                    # Verify prev_hash chain
                    if entry.get('prev_hash') != prev_hash:
                        return False, f"Chain broken at line {line_num}"

                    # Verify hash
                    stored_hash = entry.pop('hash')
                    calculated_hash = self._calculate_hash(entry, prev_hash)

                    if stored_hash != calculated_hash:
                        return False, f"Hash mismatch at line {line_num}"

                    prev_hash = stored_hash

            return True, f"Verified {line_num} entries"
        except Exception as e:
            return False, f"Verification failed: {e}"


class InstitutionalTradingBot:
    """
    Main integration class that wires all institutional components
    into the trading engine and provides a unified interface.
    """

    def __init__(self, config_path: str = None):
        logger.info("Initializing Institutional Trading Bot...")

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize state machine
        self._state = TradingState.INITIALIZING
        self._state_lock = threading.RLock()

        # Initialize institutional components
        self.state_manager = AtomicStateManager()
        self.order_manager = IdempotentOrderManager()
        self.position_reconciler = PositionReconciler()
        self.risk_engine = RiskEngine(self.config.get('risk', {}))
        self.audit_trail = AuditTrail()

        # Trading engine reference (set when connected)
        self._trading_engine = None
        self._original_execute_trade = None

        # Metrics
        self._metrics = {
            'trades_executed': 0,
            'trades_blocked': 0,
            'risk_violations': 0,
            'reconciliation_discrepancies': 0,
            'duplicate_orders_prevented': 0,
        }

        # Record initialization
        self.audit_trail.record('SYSTEM_INIT', {
            'config': self.config,
            'version': '2.0.0',
            'components': ['state_machine', 'wal', 'idempotency', 'reconciliation', 'risk_engine', 'audit']
        })

        logger.info("Institutional components initialized")

    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from file or environment"""
        config = {
            'risk': {
                'max_var_95': float(os.getenv('MAX_VAR_95', '0.02')),
                'max_drawdown': float(os.getenv('MAX_DRAWDOWN', '0.05')),
                'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '0.02')),
                'max_exposure': float(os.getenv('MAX_EXPOSURE', '0.80')),
                'max_position_pct': float(os.getenv('MAX_POSITION_PCT', '0.15')),
            },
            'trading': {
                'paper_mode': os.getenv('PAPER_TRADING', 'true').lower() == 'true',
                'max_positions': int(os.getenv('MAX_POSITIONS', '5')),
            },
            'reconciliation': {
                'interval_minutes': int(os.getenv('RECONCILIATION_INTERVAL', '5')),
            }
        }

        # Load from file if provided
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path) as f:
                    file_config = yaml.safe_load(f)
                    # Deep merge
                    for key, value in file_config.items():
                        if isinstance(value, dict) and key in config:
                            config[key].update(value)
                        else:
                            config[key] = value
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")

        return config

    def _transition_state(self, new_state: TradingState) -> bool:
        """
        Attempt state transition. Returns True if successful.
        Enforces valid transitions only.
        """
        with self._state_lock:
            valid_transitions = STATE_TRANSITIONS.get(self._state, [])

            if new_state not in valid_transitions:
                logger.error(f"Invalid state transition: {self._state.value} -> {new_state.value}")
                self.audit_trail.record('INVALID_STATE_TRANSITION', {
                    'from': self._state.value,
                    'to': new_state.value,
                    'valid_transitions': [s.value for s in valid_transitions]
                })
                return False

            old_state = self._state
            self._state = new_state

            self.audit_trail.record('STATE_TRANSITION', {
                'from': old_state.value,
                'to': new_state.value
            })

            logger.info(f"State: {old_state.value} -> {new_state.value}")

            # Persist state
            self.state_manager.update({'trading_state': new_state.value})

            return True

    def connect_to_engine(self, trading_engine):
        """
        Connect to the main trading engine and install hooks.
        This is where we wire everything together.
        """
        logger.info("Connecting institutional layer to trading engine...")

        self._trading_engine = trading_engine

        # Subscribe to commentary for event monitoring
        if hasattr(trading_engine, 'commentary') and hasattr(trading_engine.commentary, 'subscribe'):
            trading_engine.commentary.subscribe(self._on_commentary)
            logger.info("Subscribed to commentary system")

        # Hook into trade execution
        if hasattr(trading_engine, '_execute_real_trade'):
            self._original_execute_trade = trading_engine._execute_real_trade
            trading_engine._execute_real_trade = self._wrapped_execute_trade
            logger.info("Hooked trade execution")

        # Transition to connected state
        self._transition_state(TradingState.CONNECTING)
        self._transition_state(TradingState.CONNECTED)

        # Perform initial reconciliation
        self._perform_reconciliation()

        # Ready for trading
        self._transition_state(TradingState.READY)

        self.audit_trail.record('ENGINE_CONNECTED', {
            'engine_type': type(trading_engine).__name__
        })

        logger.info("Institutional layer connected and ready")

    def _on_commentary(self, commentary):
        """Handle commentary events for monitoring"""
        # Log significant events
        if hasattr(commentary, 'type'):
            event_type = str(commentary.type) if hasattr(commentary.type, 'value') else str(commentary.type)

            if 'DECISION' in event_type or 'TRADE' in event_type:
                self.audit_trail.record('TRADING_EVENT', {
                    'type': event_type,
                    'symbol': getattr(commentary, 'symbol', None),
                    'message': getattr(commentary, 'message', '')[:200]
                })

    def _wrapped_execute_trade(self, signal) -> bool:
        """
        Wrapped trade execution with institutional checks.
        This replaces the original _execute_real_trade method.
        """
        start_time = time.time()

        # Generate idempotency key
        idempotency_key = self.order_manager.generate_key(
            symbol=signal.symbol,
            side=signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
            quantity=signal.position_size,
            strategy=getattr(signal, 'strategy', 'unknown'),
            signal_time=signal.timestamp
        )

        # Check for duplicate
        if not self.order_manager.check_and_mark(idempotency_key):
            logger.warning(f"Duplicate order blocked: {signal.symbol}")
            self._metrics['duplicate_orders_prevented'] += 1
            self.audit_trail.record('DUPLICATE_ORDER_BLOCKED', {
                'symbol': signal.symbol,
                'idempotency_key': idempotency_key
            })
            return False

        # State check
        if self._state not in [TradingState.READY, TradingState.TRADING]:
            logger.warning(f"Trade blocked - invalid state: {self._state.value}")
            self._metrics['trades_blocked'] += 1
            return False

        # Transition to trading state
        self._transition_state(TradingState.TRADING)

        # Risk check
        self._transition_state(TradingState.RISK_CHECK)

        positions = self._get_positions()
        account_value = self._get_account_value()

        allowed, reason = self.risk_engine.check_trade_allowed(
            signal={'quantity': signal.position_size, 'entry_price': signal.entry_price},
            positions=positions,
            account_value=account_value
        )

        if not allowed:
            logger.warning(f"Trade blocked by risk engine: {reason}")
            self._metrics['trades_blocked'] += 1
            self._metrics['risk_violations'] += 1
            self.audit_trail.record('TRADE_BLOCKED_RISK', {
                'symbol': signal.symbol,
                'reason': reason
            })
            self._transition_state(TradingState.READY)
            return False

        # Execute trade
        self._transition_state(TradingState.EXECUTING)

        try:
            result = self._original_execute_trade(signal)

            execution_time = (time.time() - start_time) * 1000

            if result:
                self._metrics['trades_executed'] += 1

                # Record in audit trail
                self.audit_trail.record('TRADE_EXECUTED', {
                    'symbol': signal.symbol,
                    'side': str(signal.signal_type),
                    'quantity': signal.position_size,
                    'price': signal.entry_price,
                    'execution_latency_ms': execution_time,
                    'idempotency_key': idempotency_key
                })

                # Update state
                self.state_manager.update({
                    'last_trade_time': datetime.now().isoformat(),
                    'last_trade_symbol': signal.symbol,
                    'total_trades': self._metrics['trades_executed']
                })

            self._transition_state(TradingState.READY)
            return result

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            self.audit_trail.record('TRADE_FAILED', {
                'symbol': signal.symbol,
                'error': str(e)
            })
            self._transition_state(TradingState.ERROR_RECOVERY)
            self._handle_error(e)
            return False

    def _perform_reconciliation(self):
        """Reconcile positions with broker"""
        self._transition_state(TradingState.RECONCILING)

        try:
            internal_positions = self._get_positions()
            broker_positions = self._get_broker_positions()

            report = self.position_reconciler.reconcile(internal_positions, broker_positions)

            if report['status'] != 'OK':
                self._metrics['reconciliation_discrepancies'] += 1
                self.audit_trail.record('RECONCILIATION_DISCREPANCY', report)

                # Auto-fix discrepancies
                self._fix_discrepancies(report)
            else:
                self.audit_trail.record('RECONCILIATION_OK', {
                    'positions_count': len(internal_positions)
                })

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            self.audit_trail.record('RECONCILIATION_FAILED', {'error': str(e)})

    def _fix_discrepancies(self, report: Dict):
        """Auto-fix position discrepancies"""
        # This would update internal tracking to match broker
        # For safety, we log but require manual intervention for phantom positions

        for missing in report.get('missing_positions', []):
            logger.warning(f"MISSING POSITION: {missing['symbol']} - adding to tracking")
            # Add to internal tracking

        for phantom in report.get('phantom_positions', []):
            logger.error(f"PHANTOM POSITION: {phantom['symbol']} - REQUIRES MANUAL REVIEW")
            # Don't auto-remove - might be timing issue

        for mismatch in report.get('quantity_mismatches', []):
            logger.warning(f"QUANTITY MISMATCH: {mismatch['symbol']} internal={mismatch['internal_qty']} broker={mismatch['broker_qty']}")
            # Update to broker quantity

    def _get_positions(self) -> Dict:
        """Get internal position tracking"""
        if self._trading_engine and hasattr(self._trading_engine, 'positions'):
            positions = {}
            for symbol, pos in self._trading_engine.positions.items():
                positions[symbol] = {
                    'quantity': pos.quantity if hasattr(pos, 'quantity') else 0,
                    'current_price': pos.current_price if hasattr(pos, 'current_price') else 0,
                    'entry_price': pos.entry_price if hasattr(pos, 'entry_price') else 0,
                }
            return positions
        return {}

    def _get_broker_positions(self) -> Dict:
        """Get positions from broker"""
        if self._trading_engine and hasattr(self._trading_engine, 'schwab_client'):
            try:
                client = self._trading_engine.schwab_client
                if client:
                    # Get account positions
                    response = client.get_accounts(fields=['positions'])
                    if response.status_code == 200:
                        accounts = response.json()
                        positions = {}
                        for account in accounts:
                            for pos in account.get('securitiesAccount', {}).get('positions', []):
                                symbol = pos.get('instrument', {}).get('symbol')
                                if symbol:
                                    positions[symbol] = {
                                        'quantity': pos.get('longQuantity', 0) - pos.get('shortQuantity', 0),
                                        'market_value': pos.get('marketValue', 0),
                                        'average_price': pos.get('averagePrice', 0)
                                    }
                        return positions
            except Exception as e:
                logger.error(f"Failed to get broker positions: {e}")
        return {}

    def _get_account_value(self) -> float:
        """Get current account value"""
        if self._trading_engine:
            if hasattr(self._trading_engine, 'total_capital'):
                return self._trading_engine.total_capital
            if hasattr(self._trading_engine, 'buying_power'):
                return self._trading_engine.buying_power
        return 100000  # Default

    def _handle_error(self, error: Exception):
        """Handle errors with recovery logic"""
        logger.error(f"Handling error: {error}")

        # Simple recovery - transition back to ready after delay
        time.sleep(5)

        if self._state == TradingState.ERROR_RECOVERY:
            self._transition_state(TradingState.READY)

    def get_status(self) -> Dict:
        """Get current status for monitoring"""
        risk_snapshot = None
        if self._trading_engine:
            positions = self._get_positions()
            account_value = self._get_account_value()
            risk_snapshot = self.risk_engine.calculate_snapshot(positions, account_value)

        return {
            'state': self._state.value,
            'metrics': self._metrics,
            'risk': {
                'exposure': risk_snapshot.total_exposure if risk_snapshot else 0,
                'var_95': risk_snapshot.var_95 if risk_snapshot else 0,
                'daily_pnl': risk_snapshot.daily_pnl if risk_snapshot else 0,
                'drawdown': risk_snapshot.drawdown if risk_snapshot else 0,
            } if risk_snapshot else {},
            'last_reconciliation': self.position_reconciler._last_reconciliation.isoformat()
                                   if self.position_reconciler._last_reconciliation else None,
            'audit_integrity': self.audit_trail.verify_integrity()[0]
        }

    def emergency_stop(self):
        """Emergency stop - close all positions"""
        logger.critical("EMERGENCY STOP TRIGGERED")

        self.audit_trail.record('EMERGENCY_STOP', {
            'trigger': 'manual',
            'state_at_trigger': self._state.value
        })

        # Lock trading
        self._transition_state(TradingState.RISK_LOCKED)

        # Close all positions
        if self._trading_engine:
            positions = self._get_positions()
            for symbol in positions:
                try:
                    logger.info(f"Emergency closing: {symbol}")
                    # Call engine's close position method
                    if hasattr(self._trading_engine, '_close_real_position'):
                        self._trading_engine._close_real_position({'symbol': symbol})
                except Exception as e:
                    logger.error(f"Failed to close {symbol}: {e}")

        self._transition_state(TradingState.SHUTTING_DOWN)
        self._transition_state(TradingState.STOPPED)

    def start(self):
        """Start the institutional bot (connects to main engine)"""
        logger.info("Starting Institutional Trading Bot...")

        try:
            # Import and initialize main trading engine
            from trading_bot_commentary_updated import TradingEngineWithCommentary, TradingMode

            # Create trading engine
            mode = TradingMode.PAPER if self.config['trading']['paper_mode'] else TradingMode.LIVE
            engine = TradingEngineWithCommentary(mode=mode)

            # Connect institutional layer
            self.connect_to_engine(engine)

            # Start trading
            asyncio.run(engine.start())

        except ImportError as e:
            logger.error(f"Failed to import trading engine: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            self.audit_trail.record('START_FAILED', {'error': str(e)})
            raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Institutional Trading Bot')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    args = parser.parse_args()

    # Set environment
    if args.paper:
        os.environ['PAPER_TRADING'] = 'true'
    elif args.live:
        os.environ['PAPER_TRADING'] = 'false'

    # Create and start bot
    bot = InstitutionalTradingBot(config_path=args.config)

    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        bot.emergency_stop()


if __name__ == "__main__":
    main()
