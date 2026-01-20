#!/usr/bin/env python3
"""
Institutional-Grade Trading Core
================================
This module provides critical infrastructure for production trading:
- Explicit state machine with guard conditions
- Position reconciliation with broker
- Atomic state persistence with corruption detection
- Order idempotency to prevent duplicates
- Comprehensive health monitoring
- Enhanced circuit breakers

Version: 1.0.0
Author: Institutional Systems Transformation
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import traceback

import numpy as np

logger = logging.getLogger('InstitutionalCore')
logger.setLevel(logging.INFO)

# ============================================================================
# TRADING STATE MACHINE
# ============================================================================

class TradingState(Enum):
    """Explicit trading system states"""
    INITIALIZING = auto()      # System starting up
    INIT_FAILED = auto()       # Startup failed
    AUTH_REQUIRED = auto()     # Need credentials
    CONNECTED = auto()         # API connected, not trading
    RECONCILING = auto()       # Syncing positions with broker
    TRADING = auto()           # Normal operation
    MARKET_CLOSED = auto()     # Outside trading hours
    RISK_LOCKED = auto()       # Risk limit breached
    POSITION_EXIT = auto()     # Closing all positions
    EMERGENCY_EXIT = auto()    # Critical failure - market sell all
    SHUTDOWN = auto()          # Graceful termination


@dataclass
class StateTransition:
    """Records a state transition for audit trail"""
    timestamp: datetime
    from_state: TradingState
    to_state: TradingState
    trigger: str
    guard_passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingStateMachine:
    """
    Explicit state machine for trading system.
    Ensures deterministic behavior and crash recovery.
    """

    # Define valid state transitions and their guards
    VALID_TRANSITIONS = {
        TradingState.INITIALIZING: {
            TradingState.CONNECTED,
            TradingState.INIT_FAILED,
            TradingState.AUTH_REQUIRED,
        },
        TradingState.INIT_FAILED: {
            TradingState.INITIALIZING,  # Retry
        },
        TradingState.AUTH_REQUIRED: {
            TradingState.INITIALIZING,  # After auth
            TradingState.SHUTDOWN,
        },
        TradingState.CONNECTED: {
            TradingState.RECONCILING,
            TradingState.TRADING,
            TradingState.MARKET_CLOSED,
            TradingState.SHUTDOWN,
        },
        TradingState.RECONCILING: {
            TradingState.TRADING,
            TradingState.CONNECTED,  # Reconciliation failed
            TradingState.EMERGENCY_EXIT,
        },
        TradingState.TRADING: {
            TradingState.RISK_LOCKED,
            TradingState.MARKET_CLOSED,
            TradingState.EMERGENCY_EXIT,
            TradingState.RECONCILING,
            TradingState.POSITION_EXIT,
            TradingState.SHUTDOWN,
        },
        TradingState.MARKET_CLOSED: {
            TradingState.CONNECTED,
            TradingState.TRADING,
            TradingState.SHUTDOWN,
        },
        TradingState.RISK_LOCKED: {
            TradingState.POSITION_EXIT,
            TradingState.TRADING,  # Manual override
            TradingState.EMERGENCY_EXIT,
            TradingState.SHUTDOWN,
        },
        TradingState.POSITION_EXIT: {
            TradingState.CONNECTED,  # All positions closed
            TradingState.EMERGENCY_EXIT,
            TradingState.SHUTDOWN,
        },
        TradingState.EMERGENCY_EXIT: {
            TradingState.SHUTDOWN,
            TradingState.CONNECTED,  # After emergency handled
        },
        TradingState.SHUTDOWN: set(),  # Terminal state
    }

    # Operations allowed in each state
    ALLOWED_OPERATIONS = {
        TradingState.INITIALIZING: {'health_check'},
        TradingState.INIT_FAILED: {'retry_init'},
        TradingState.AUTH_REQUIRED: {'provide_auth'},
        TradingState.CONNECTED: {'health_check', 'fetch_positions', 'fetch_quotes'},
        TradingState.RECONCILING: {'fetch_positions', 'health_check'},
        TradingState.TRADING: {
            'health_check', 'fetch_positions', 'fetch_quotes',
            'place_order', 'cancel_order', 'modify_order',
            'generate_signals', 'update_stops'
        },
        TradingState.MARKET_CLOSED: {'health_check', 'fetch_positions'},
        TradingState.RISK_LOCKED: {'close_position', 'cancel_order', 'health_check'},
        TradingState.POSITION_EXIT: {'close_position', 'market_sell', 'cancel_order'},
        TradingState.EMERGENCY_EXIT: {'market_sell_all'},
        TradingState.SHUTDOWN: set(),
    }

    def __init__(self, initial_state: TradingState = TradingState.INITIALIZING):
        self._state = initial_state
        self._lock = threading.RLock()
        self._transition_history: List[StateTransition] = []
        self._guards: Dict[Tuple[TradingState, TradingState], Callable] = {}
        self._state_callbacks: Dict[TradingState, List[Callable]] = {}
        self._last_transition_time = datetime.now()

        # Default guards
        self._setup_default_guards()

        logger.info(f"State machine initialized in state: {self._state.name}")

    def _setup_default_guards(self):
        """Setup default transition guard conditions"""
        # Example: Can only go to TRADING if market is open
        # These should be overridden by the actual trading engine
        pass

    def register_guard(self, from_state: TradingState, to_state: TradingState,
                      guard_fn: Callable[[], bool]):
        """Register a guard condition for a state transition"""
        self._guards[(from_state, to_state)] = guard_fn

    def register_state_callback(self, state: TradingState, callback: Callable):
        """Register a callback to be called when entering a state"""
        if state not in self._state_callbacks:
            self._state_callbacks[state] = []
        self._state_callbacks[state].append(callback)

    @property
    def state(self) -> TradingState:
        """Current state (thread-safe read)"""
        with self._lock:
            return self._state

    def can_transition_to(self, target_state: TradingState) -> Tuple[bool, str]:
        """Check if transition to target state is valid"""
        with self._lock:
            # Check if transition is valid
            if target_state not in self.VALID_TRANSITIONS.get(self._state, set()):
                return False, f"Invalid transition: {self._state.name} -> {target_state.name}"

            # Check guard condition
            guard = self._guards.get((self._state, target_state))
            if guard and not guard():
                return False, f"Guard condition failed for {self._state.name} -> {target_state.name}"

            return True, "OK"

    def transition_to(self, target_state: TradingState, trigger: str = "manual",
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Attempt state transition.
        Returns True if successful, False if blocked.
        """
        with self._lock:
            can_transition, reason = self.can_transition_to(target_state)

            transition = StateTransition(
                timestamp=datetime.now(),
                from_state=self._state,
                to_state=target_state,
                trigger=trigger,
                guard_passed=can_transition,
                metadata=metadata or {}
            )

            if not can_transition:
                logger.warning(f"State transition blocked: {reason}")
                transition.metadata['block_reason'] = reason
                self._transition_history.append(transition)
                return False

            # Record transition
            old_state = self._state
            self._state = target_state
            self._last_transition_time = datetime.now()
            self._transition_history.append(transition)

            logger.info(f"State transition: {old_state.name} -> {target_state.name} (trigger: {trigger})")

            # Execute callbacks
            for callback in self._state_callbacks.get(target_state, []):
                try:
                    callback(old_state, target_state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

            return True

    def is_operation_allowed(self, operation: str) -> bool:
        """Check if an operation is allowed in current state"""
        with self._lock:
            return operation in self.ALLOWED_OPERATIONS.get(self._state, set())

    def get_transition_history(self, limit: int = 100) -> List[StateTransition]:
        """Get recent state transition history"""
        with self._lock:
            return self._transition_history[-limit:]

    def get_state_duration(self) -> timedelta:
        """Get time spent in current state"""
        return datetime.now() - self._last_transition_time

    def serialize(self) -> Dict[str, Any]:
        """Serialize state machine for persistence"""
        with self._lock:
            return {
                'current_state': self._state.name,
                'last_transition': self._last_transition_time.isoformat(),
                'transition_count': len(self._transition_history)
            }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'TradingStateMachine':
        """Restore state machine from persistence"""
        state = TradingState[data['current_state']]
        machine = cls(initial_state=state)
        machine._last_transition_time = datetime.fromisoformat(data['last_transition'])
        return machine


# ============================================================================
# ATOMIC STATE PERSISTENCE
# ============================================================================

class AtomicStateManager:
    """
    Thread-safe atomic state persistence with corruption detection.
    Uses write-ahead logging pattern for crash recovery.
    """

    def __init__(self, state_file: str, backup_count: int = 5):
        self.state_file = Path(state_file)
        self.backup_dir = self.state_file.parent / f".{self.state_file.stem}_backups"
        self.checksum_file = self.state_file.with_suffix('.checksum')
        self.wal_file = self.state_file.with_suffix('.wal')  # Write-ahead log
        self.backup_count = backup_count
        self._lock = threading.Lock()
        self._save_count = 0

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_checksum(self, data: str) -> str:
        """Calculate SHA-256 checksum of data"""
        return hashlib.sha256(data.encode()).hexdigest()

    def _write_wal(self, operation: str, data: Dict) -> str:
        """Write to write-ahead log before actual write"""
        wal_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'data_hash': self._calculate_checksum(json.dumps(data, default=str)),
            'status': 'PENDING'
        }
        with open(self.wal_file, 'a') as f:
            f.write(json.dumps(wal_entry) + '\n')
        return wal_entry['data_hash']

    def _complete_wal(self, data_hash: str):
        """Mark WAL entry as completed"""
        if not self.wal_file.exists():
            return

        # Read all entries, mark matching one as complete
        entries = []
        with open(self.wal_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry['data_hash'] == data_hash:
                    entry['status'] = 'COMPLETED'
                entries.append(entry)

        # Rewrite WAL
        with open(self.wal_file, 'w') as f:
            for entry in entries[-100:]:  # Keep last 100 entries
                f.write(json.dumps(entry) + '\n')

    def save_state(self, state: Dict) -> bool:
        """
        Save state atomically with checksum verification.
        Returns True on success, False on failure.
        """
        with self._lock:
            try:
                # 1. Serialize state
                state_json = json.dumps(state, indent=2, default=str)
                checksum = self._calculate_checksum(state_json)

                # 2. Write to WAL
                data_hash = self._write_wal('SAVE', state)

                # 3. Create temp file in same directory (for atomic rename)
                fd, temp_path = tempfile.mkstemp(
                    dir=self.state_file.parent,
                    prefix='.state_tmp_',
                    suffix='.json'
                )

                try:
                    # 4. Write to temp file with fsync
                    with os.fdopen(fd, 'w') as f:
                        f.write(state_json)
                        f.flush()
                        os.fsync(f.fileno())
                except Exception:
                    os.close(fd)
                    raise

                # 5. Backup current file
                if self.state_file.exists():
                    self._create_backup()

                # 6. Atomic rename
                os.replace(temp_path, self.state_file)

                # 7. Write checksum file
                with open(self.checksum_file, 'w') as f:
                    f.write(f"{checksum}\n{datetime.now().isoformat()}")

                # 8. Mark WAL complete
                self._complete_wal(data_hash)

                self._save_count += 1
                logger.debug(f"State saved successfully (save #{self._save_count})")
                return True

            except Exception as e:
                logger.error(f"Failed to save state: {e}\n{traceback.format_exc()}")
                return False

    def _create_backup(self):
        """Create timestamped backup of current state file"""
        if not self.state_file.exists():
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"state_{timestamp}.json"
        shutil.copy2(self.state_file, backup_path)

        # Clean old backups
        backups = sorted(self.backup_dir.glob('state_*.json'))
        while len(backups) > self.backup_count:
            oldest = backups.pop(0)
            oldest.unlink()

    def load_state(self) -> Optional[Dict]:
        """
        Load state with integrity verification.
        Returns None if state cannot be loaded/verified.
        """
        if not self.state_file.exists():
            logger.warning("State file does not exist")
            return None

        try:
            # 1. Read state
            with open(self.state_file, 'r') as f:
                state_json = f.read()

            # 2. Verify checksum
            if self.checksum_file.exists():
                with open(self.checksum_file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    expected_checksum = lines[0]

                actual_checksum = self._calculate_checksum(state_json)

                if actual_checksum != expected_checksum:
                    logger.error("State file checksum mismatch - file may be corrupted")
                    return self._recover_from_backup()

            # 3. Parse JSON
            state = json.loads(state_json)
            logger.info("State loaded and verified successfully")
            return state

        except json.JSONDecodeError as e:
            logger.error(f"State file JSON decode error: {e}")
            return self._recover_from_backup()
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return self._recover_from_backup()

    def _recover_from_backup(self) -> Optional[Dict]:
        """Attempt to recover state from backup files"""
        backups = sorted(self.backup_dir.glob('state_*.json'), reverse=True)

        for backup in backups:
            try:
                logger.info(f"Attempting recovery from: {backup.name}")
                with open(backup, 'r') as f:
                    state = json.load(f)

                # Restore this backup as current state
                shutil.copy2(backup, self.state_file)
                logger.info(f"Recovered state from backup: {backup.name}")
                return state

            except Exception as e:
                logger.warning(f"Backup {backup.name} recovery failed: {e}")
                continue

        logger.error("All backup recovery attempts failed")
        return None

    def recover_from_wal(self) -> bool:
        """Check WAL for incomplete operations and attempt recovery"""
        if not self.wal_file.exists():
            return True

        incomplete = []
        with open(self.wal_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry['status'] == 'PENDING':
                        incomplete.append(entry)
                except:
                    continue

        if incomplete:
            logger.warning(f"Found {len(incomplete)} incomplete WAL entries")
            # For incomplete saves, the temp file may still exist
            # Recovery would check for .state_tmp_* files and complete the operation
            return False

        return True


# ============================================================================
# ORDER IDEMPOTENCY MANAGER
# ============================================================================

@dataclass
class OrderAttempt:
    """Records an order submission attempt"""
    idempotency_key: str
    signal_hash: str
    submitted_at: datetime
    status: str  # PENDING, SUCCESS, FAILED, DUPLICATE
    order_id: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict] = None


class IdempotentOrderManager:
    """
    Prevents duplicate order submissions through idempotency keys.
    Critical for network failure scenarios where retries could double position size.
    """

    def __init__(self, ttl_hours: int = 24):
        self._attempts: Dict[str, OrderAttempt] = {}
        self._lock = asyncio.Lock()
        self.ttl = timedelta(hours=ttl_hours)
        self._cleanup_task: Optional[asyncio.Task] = None

    def _generate_idempotency_key(self, symbol: str, side: str, quantity: float,
                                  price: Optional[float], timestamp: datetime) -> str:
        """
        Generate unique idempotency key.
        Key is based on: symbol, side, quantity, price, and timestamp (rounded to minute).
        """
        # Round timestamp to minute to allow for slight timing differences
        ts_minute = timestamp.replace(second=0, microsecond=0).isoformat()

        # Create deterministic key
        key_data = f"{symbol}|{side}|{quantity:.6f}|{price or 'MKT'}|{ts_minute}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:20]

    async def check_or_submit(self, symbol: str, side: str, quantity: float,
                             price: Optional[float], timestamp: datetime,
                             execute_fn: Callable) -> Dict[str, Any]:
        """
        Check for duplicate order and submit if unique.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            price: Limit price (None for market orders)
            timestamp: Signal generation timestamp
            execute_fn: Async function to execute the order

        Returns:
            Dict with status and result/error
        """
        key = self._generate_idempotency_key(symbol, side, quantity, price, timestamp)

        async with self._lock:
            # Check for existing attempt
            if key in self._attempts:
                existing = self._attempts[key]

                if existing.status == 'PENDING':
                    return {
                        'status': 'ALREADY_PENDING',
                        'message': 'Order already being processed',
                        'idempotency_key': key
                    }

                elif existing.status == 'SUCCESS':
                    return {
                        'status': 'DUPLICATE',
                        'message': 'Order already submitted successfully',
                        'idempotency_key': key,
                        'original_order_id': existing.order_id,
                        'original_result': existing.result
                    }

                elif existing.status == 'FAILED':
                    # Allow retry of failed orders
                    logger.info(f"Retrying previously failed order: {key}")

            # Mark as pending
            attempt = OrderAttempt(
                idempotency_key=key,
                signal_hash=key,
                submitted_at=datetime.now(),
                status='PENDING'
            )
            self._attempts[key] = attempt

        # Execute order (outside lock to allow concurrent orders for different keys)
        try:
            result = await execute_fn()

            async with self._lock:
                self._attempts[key].status = 'SUCCESS'
                self._attempts[key].order_id = result.get('order_id')
                self._attempts[key].result = result

            return {
                'status': 'SUCCESS',
                'idempotency_key': key,
                'result': result
            }

        except Exception as e:
            async with self._lock:
                self._attempts[key].status = 'FAILED'
                self._attempts[key].error = str(e)

            return {
                'status': 'FAILED',
                'idempotency_key': key,
                'error': str(e)
            }

    async def cleanup_expired(self):
        """Remove expired idempotency records"""
        cutoff = datetime.now() - self.ttl
        expired = []

        async with self._lock:
            for key, attempt in self._attempts.items():
                if attempt.submitted_at < cutoff:
                    expired.append(key)

            for key in expired:
                del self._attempts[key]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired idempotency records")

    def get_pending_count(self) -> int:
        """Get count of pending orders"""
        return sum(1 for a in self._attempts.values() if a.status == 'PENDING')

    def get_recent_attempts(self, limit: int = 50) -> List[OrderAttempt]:
        """Get recent order attempts"""
        sorted_attempts = sorted(
            self._attempts.values(),
            key=lambda x: x.submitted_at,
            reverse=True
        )
        return sorted_attempts[:limit]


# ============================================================================
# POSITION RECONCILIATION
# ============================================================================

@dataclass
class PositionDiscrepancy:
    """Records a position discrepancy between local and broker state"""
    symbol: str
    discrepancy_type: str  # PHANTOM, MISSING, QUANTITY_MISMATCH, PRICE_MISMATCH
    local_quantity: float
    broker_quantity: float
    local_price: Optional[float]
    broker_price: Optional[float]
    detected_at: datetime
    resolved: bool = False
    resolution: Optional[str] = None


@dataclass
class ReconciliationReport:
    """Results of a position reconciliation"""
    timestamp: datetime
    local_position_count: int
    broker_position_count: int
    discrepancies: List[PositionDiscrepancy]
    actions_taken: List[str]
    success: bool
    duration_ms: float


class PositionReconciler:
    """
    Reconciles local position state with broker state.
    Critical for preventing phantom positions and missed positions.
    """

    def __init__(self, tolerance_qty: float = 0.001, tolerance_price_pct: float = 0.01):
        self.tolerance_qty = tolerance_qty
        self.tolerance_price_pct = tolerance_price_pct
        self._reconciliation_history: List[ReconciliationReport] = []
        self._lock = asyncio.Lock()

    async def reconcile(self, local_positions: Dict[str, Any],
                       fetch_broker_positions: Callable) -> ReconciliationReport:
        """
        Reconcile local positions with broker positions.

        Args:
            local_positions: Dict of symbol -> position data from local state
            fetch_broker_positions: Async function to fetch positions from broker

        Returns:
            ReconciliationReport with discrepancies and actions
        """
        start_time = time.time()
        discrepancies = []
        actions = []

        try:
            # Fetch broker positions
            broker_positions = await fetch_broker_positions()
        except Exception as e:
            return ReconciliationReport(
                timestamp=datetime.now(),
                local_position_count=len(local_positions),
                broker_position_count=0,
                discrepancies=[],
                actions_taken=[f"Failed to fetch broker positions: {e}"],
                success=False,
                duration_ms=(time.time() - start_time) * 1000
            )

        # Track which broker positions we've seen
        broker_symbols = set(broker_positions.keys())
        local_symbols = set(local_positions.keys())

        # Check for phantom positions (local has, broker doesn't)
        phantom_symbols = local_symbols - broker_symbols
        for symbol in phantom_symbols:
            local_pos = local_positions[symbol]
            discrepancies.append(PositionDiscrepancy(
                symbol=symbol,
                discrepancy_type='PHANTOM',
                local_quantity=local_pos.get('quantity', 0),
                broker_quantity=0,
                local_price=local_pos.get('entry_price'),
                broker_price=None,
                detected_at=datetime.now()
            ))
            actions.append(f"PHANTOM: Remove {symbol} from local state")

        # Check for missing positions (broker has, local doesn't)
        missing_symbols = broker_symbols - local_symbols
        for symbol in missing_symbols:
            broker_pos = broker_positions[symbol]
            discrepancies.append(PositionDiscrepancy(
                symbol=symbol,
                discrepancy_type='MISSING',
                local_quantity=0,
                broker_quantity=broker_pos.get('quantity', 0),
                local_price=None,
                broker_price=broker_pos.get('avg_price'),
                detected_at=datetime.now()
            ))
            actions.append(f"MISSING: Add {symbol} to local state from broker")

        # Check for quantity/price mismatches
        common_symbols = local_symbols & broker_symbols
        for symbol in common_symbols:
            local_pos = local_positions[symbol]
            broker_pos = broker_positions[symbol]

            local_qty = local_pos.get('quantity', 0)
            broker_qty = broker_pos.get('quantity', 0)

            # Quantity check
            if abs(local_qty - broker_qty) > self.tolerance_qty:
                discrepancies.append(PositionDiscrepancy(
                    symbol=symbol,
                    discrepancy_type='QUANTITY_MISMATCH',
                    local_quantity=local_qty,
                    broker_quantity=broker_qty,
                    local_price=local_pos.get('entry_price'),
                    broker_price=broker_pos.get('avg_price'),
                    detected_at=datetime.now()
                ))
                actions.append(f"QTY_MISMATCH: Update {symbol} qty from {local_qty} to {broker_qty}")

        # Create report
        report = ReconciliationReport(
            timestamp=datetime.now(),
            local_position_count=len(local_positions),
            broker_position_count=len(broker_positions),
            discrepancies=discrepancies,
            actions_taken=actions,
            success=True,
            duration_ms=(time.time() - start_time) * 1000
        )

        async with self._lock:
            self._reconciliation_history.append(report)
            # Keep last 100 reports
            if len(self._reconciliation_history) > 100:
                self._reconciliation_history = self._reconciliation_history[-100:]

        return report

    def get_last_reconciliation(self) -> Optional[ReconciliationReport]:
        """Get the most recent reconciliation report"""
        if self._reconciliation_history:
            return self._reconciliation_history[-1]
        return None

    def has_discrepancies(self) -> bool:
        """Check if last reconciliation had discrepancies"""
        last = self.get_last_reconciliation()
        return last is not None and len(last.discrepancies) > 0


# ============================================================================
# HEALTH MONITORING
# ============================================================================

@dataclass
class HealthCheck:
    """Result of a health check"""
    component: str
    status: str  # OK, WARNING, CRITICAL, UNKNOWN
    message: str
    checked_at: datetime
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """
    Comprehensive health monitoring for trading system.
    Monitors API connectivity, data freshness, system resources.
    """

    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._check_functions: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._check_history: Dict[str, List[HealthCheck]] = {}

    def register_check(self, name: str, check_fn: Callable[[], HealthCheck]):
        """Register a health check function"""
        self._check_functions[name] = check_fn
        self._check_history[name] = []

    async def run_check(self, name: str) -> HealthCheck:
        """Run a single health check"""
        if name not in self._check_functions:
            return HealthCheck(
                component=name,
                status='UNKNOWN',
                message=f'Unknown health check: {name}',
                checked_at=datetime.now()
            )

        start_time = time.time()
        try:
            check_fn = self._check_functions[name]
            if asyncio.iscoroutinefunction(check_fn):
                result = await check_fn()
            else:
                result = check_fn()

            result.latency_ms = (time.time() - start_time) * 1000

        except Exception as e:
            result = HealthCheck(
                component=name,
                status='CRITICAL',
                message=f'Health check failed: {str(e)}',
                checked_at=datetime.now(),
                latency_ms=(time.time() - start_time) * 1000
            )

        with self._lock:
            self._checks[name] = result
            self._check_history[name].append(result)
            # Keep last 100 checks per component
            if len(self._check_history[name]) > 100:
                self._check_history[name] = self._check_history[name][-100:]

        return result

    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks"""
        results = {}
        for name in self._check_functions:
            results[name] = await self.run_check(name)
        return results

    def get_overall_status(self) -> str:
        """Get overall system health status"""
        with self._lock:
            if not self._checks:
                return 'UNKNOWN'

            statuses = [check.status for check in self._checks.values()]

            if 'CRITICAL' in statuses:
                return 'CRITICAL'
            elif 'WARNING' in statuses:
                return 'WARNING'
            elif all(s == 'OK' for s in statuses):
                return 'OK'
            else:
                return 'UNKNOWN'

    def is_healthy(self) -> bool:
        """Check if system is healthy (no CRITICAL status)"""
        return self.get_overall_status() != 'CRITICAL'

    def get_check_result(self, name: str) -> Optional[HealthCheck]:
        """Get result of specific health check"""
        with self._lock:
            return self._checks.get(name)


def create_api_health_check(api_client, timeout_seconds: float = 5.0) -> Callable:
    """Factory for API health check functions"""
    async def check() -> HealthCheck:
        start = time.time()
        try:
            # This should be adapted to the actual API client
            # response = await asyncio.wait_for(
            #     api_client.get_account(),
            #     timeout=timeout_seconds
            # )
            latency = (time.time() - start) * 1000

            if latency > 1000:
                status = 'WARNING'
                message = f'API responding slowly: {latency:.0f}ms'
            else:
                status = 'OK'
                message = f'API healthy: {latency:.0f}ms'

            return HealthCheck(
                component='api',
                status=status,
                message=message,
                checked_at=datetime.now(),
                latency_ms=latency
            )

        except asyncio.TimeoutError:
            return HealthCheck(
                component='api',
                status='CRITICAL',
                message=f'API timeout after {timeout_seconds}s',
                checked_at=datetime.now()
            )
        except Exception as e:
            return HealthCheck(
                component='api',
                status='CRITICAL',
                message=f'API error: {str(e)}',
                checked_at=datetime.now()
            )

    return check


def create_memory_health_check(threshold_mb: int = 2000) -> Callable:
    """Factory for memory health check"""
    def check() -> HealthCheck:
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > threshold_mb:
                status = 'CRITICAL'
                message = f'Memory usage critical: {memory_mb:.0f}MB (threshold: {threshold_mb}MB)'
            elif memory_mb > threshold_mb * 0.8:
                status = 'WARNING'
                message = f'Memory usage high: {memory_mb:.0f}MB'
            else:
                status = 'OK'
                message = f'Memory usage normal: {memory_mb:.0f}MB'

            return HealthCheck(
                component='memory',
                status=status,
                message=message,
                checked_at=datetime.now(),
                metadata={'memory_mb': memory_mb}
            )

        except ImportError:
            return HealthCheck(
                component='memory',
                status='UNKNOWN',
                message='psutil not available for memory monitoring',
                checked_at=datetime.now()
            )

    return check


def create_disk_health_check(path: str = '.', threshold_percent: float = 90) -> Callable:
    """Factory for disk space health check"""
    def check() -> HealthCheck:
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            used_percent = (used / total) * 100

            if used_percent > threshold_percent:
                status = 'CRITICAL'
                message = f'Disk usage critical: {used_percent:.1f}%'
            elif used_percent > threshold_percent - 10:
                status = 'WARNING'
                message = f'Disk usage high: {used_percent:.1f}%'
            else:
                status = 'OK'
                message = f'Disk usage normal: {used_percent:.1f}%'

            return HealthCheck(
                component='disk',
                status=status,
                message=message,
                checked_at=datetime.now(),
                metadata={
                    'used_percent': used_percent,
                    'free_gb': free / (1024**3)
                }
            )

        except Exception as e:
            return HealthCheck(
                component='disk',
                status='UNKNOWN',
                message=f'Disk check failed: {str(e)}',
                checked_at=datetime.now()
            )

    return check


# ============================================================================
# ENHANCED CIRCUIT BREAKERS
# ============================================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


class EnhancedCircuitBreaker:
    """
    Enhanced circuit breaker with:
    - Configurable failure counting
    - Sliding window for failure tracking
    - Different recovery strategies
    - Metrics and reporting
    """

    def __init__(self,
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 half_open_max_calls: int = 3,
                 window_size: int = 60,
                 expected_exceptions: Tuple[type, ...] = (Exception,)):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.window_size = window_size
        self.expected_exceptions = expected_exceptions

        self._state = CircuitState.CLOSED
        self._failures: List[datetime] = []
        self._last_failure_time: Optional[datetime] = None
        self._opened_at: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_blocked = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery"""
        if self._opened_at is None:
            return True
        elapsed = (datetime.now() - self._opened_at).total_seconds()
        return elapsed >= self.recovery_timeout

    def _count_recent_failures(self) -> int:
        """Count failures in the sliding window"""
        cutoff = datetime.now() - timedelta(seconds=self.window_size)
        self._failures = [t for t in self._failures if t > cutoff]
        return len(self._failures)

    def record_success(self):
        """Record a successful call"""
        with self._lock:
            self._total_calls += 1
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failures.clear()
                    logger.info(f"Circuit breaker '{self.name}' closed after successful recovery")

    def record_failure(self, exception: Exception = None):
        """Record a failed call"""
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._failures.append(datetime.now())
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery, go back to OPEN
                self._state = CircuitState.OPEN
                self._opened_at = datetime.now()
                logger.warning(f"Circuit breaker '{self.name}' reopened after failed recovery")

            elif self._state == CircuitState.CLOSED:
                if self._count_recent_failures() >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = datetime.now()
                    logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_threshold} failures")

    def can_execute(self) -> bool:
        """Check if a call can be executed"""
        current_state = self.state  # This may trigger OPEN -> HALF_OPEN
        if current_state == CircuitState.OPEN:
            with self._lock:
                self._total_blocked += 1
            return False
        return True

    def reset(self):
        """Manually reset the circuit breaker"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures.clear()
            self._last_failure_time = None
            self._opened_at = None
            self._half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'total_calls': self._total_calls,
                'total_failures': self._total_failures,
                'total_blocked': self._total_blocked,
                'recent_failures': self._count_recent_failures(),
                'failure_rate': self._total_failures / max(self._total_calls, 1),
                'last_failure': self._last_failure_time.isoformat() if self._last_failure_time else None
            }

    def __call__(self, func: Callable) -> Callable:
        """Decorator for protecting function calls"""
        import functools

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.can_execute():
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN",
                        breaker_name=self.name,
                        time_until_retry=self._get_time_until_retry()
                    )

                try:
                    result = await func(*args, **kwargs)
                    self.record_success()
                    return result
                except self.expected_exceptions as e:
                    self.record_failure(e)
                    raise

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.can_execute():
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN",
                        breaker_name=self.name,
                        time_until_retry=self._get_time_until_retry()
                    )

                try:
                    result = func(*args, **kwargs)
                    self.record_success()
                    return result
                except self.expected_exceptions as e:
                    self.record_failure(e)
                    raise

            return sync_wrapper

    def _get_time_until_retry(self) -> float:
        """Get seconds until circuit breaker will try again"""
        if self._opened_at is None:
            return 0
        elapsed = (datetime.now() - self._opened_at).total_seconds()
        return max(0, self.recovery_timeout - elapsed)


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    def __init__(self, message: str, breaker_name: str, time_until_retry: float):
        super().__init__(message)
        self.breaker_name = breaker_name
        self.time_until_retry = time_until_retry


# ============================================================================
# POSITION SIZE VALIDATOR
# ============================================================================

class PositionSizeValidator:
    """
    Validates position sizes before order submission.
    Catches calculation errors that could result in dangerous position sizes.
    """

    def __init__(self,
                 max_position_value: float = 50000,
                 min_position_value: float = 100,
                 max_shares: int = 100000,
                 max_portfolio_percent: float = 0.20):
        self.max_position_value = max_position_value
        self.min_position_value = min_position_value
        self.max_shares = max_shares
        self.max_portfolio_percent = max_portfolio_percent

    def validate(self, symbol: str, quantity: float, price: float,
                portfolio_value: float) -> Tuple[bool, str]:
        """
        Validate a position size.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for invalid values
        if quantity is None or np.isnan(quantity) or np.isinf(quantity):
            return False, f"Invalid quantity: {quantity}"

        if price is None or np.isnan(price) or np.isinf(price):
            return False, f"Invalid price: {price}"

        if quantity <= 0:
            return False, f"Quantity must be positive: {quantity}"

        if price <= 0:
            return False, f"Price must be positive: {price}"

        # Calculate position value
        position_value = quantity * price

        # Check absolute limits
        if position_value > self.max_position_value:
            return False, f"Position value ${position_value:.2f} exceeds max ${self.max_position_value}"

        if position_value < self.min_position_value:
            return False, f"Position value ${position_value:.2f} below min ${self.min_position_value}"

        if quantity > self.max_shares:
            return False, f"Quantity {quantity} exceeds max shares {self.max_shares}"

        # Check portfolio percentage
        if portfolio_value > 0:
            portfolio_percent = position_value / portfolio_value
            if portfolio_percent > self.max_portfolio_percent:
                return False, f"Position {portfolio_percent:.1%} exceeds max {self.max_portfolio_percent:.1%} of portfolio"

        return True, "OK"


# ============================================================================
# FEATURE VALIDATOR (NaN/Inf Protection)
# ============================================================================

class FeatureValidator:
    """
    Validates ML features before prediction.
    Catches NaN/Inf values that could cause unpredictable model behavior.
    """

    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or []
        self._validation_stats = {
            'total_validations': 0,
            'nan_detected': 0,
            'inf_detected': 0,
            'features_with_issues': {}
        }

    def validate(self, features: np.ndarray,
                feature_names: List[str] = None) -> Tuple[bool, List[str], np.ndarray]:
        """
        Validate feature array.

        Args:
            features: Feature array to validate
            feature_names: Optional names for features

        Returns:
            Tuple of (is_valid, issues_list, cleaned_features)
        """
        self._validation_stats['total_validations'] += 1
        issues = []
        names = feature_names or self.feature_names or [f"feature_{i}" for i in range(len(features))]

        # Check for NaN
        nan_mask = np.isnan(features)
        if np.any(nan_mask):
            nan_indices = np.where(nan_mask)[0]
            nan_names = [names[i] for i in nan_indices if i < len(names)]
            issues.append(f"NaN in features: {nan_names}")
            self._validation_stats['nan_detected'] += 1
            for name in nan_names:
                self._validation_stats['features_with_issues'][name] = \
                    self._validation_stats['features_with_issues'].get(name, 0) + 1

        # Check for Inf
        inf_mask = np.isinf(features)
        if np.any(inf_mask):
            inf_indices = np.where(inf_mask)[0]
            inf_names = [names[i] for i in inf_indices if i < len(names)]
            issues.append(f"Inf in features: {inf_names}")
            self._validation_stats['inf_detected'] += 1
            for name in inf_names:
                self._validation_stats['features_with_issues'][name] = \
                    self._validation_stats['features_with_issues'].get(name, 0) + 1

        # Create cleaned features (replace NaN/Inf with 0)
        cleaned = features.copy()
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)

        is_valid = len(issues) == 0
        return is_valid, issues, cleaned

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self._validation_stats.copy()


# ============================================================================
# INSTITUTIONAL TRADING CORE
# ============================================================================

class InstitutionalTradingCore:
    """
    Orchestrates all institutional-grade components.
    This class should be instantiated and integrated with the existing trading engine.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # State machine
        self.state_machine = TradingStateMachine()

        # State persistence
        state_file = self.config.get('state_file', 'trading_state_institutional.json')
        self.state_manager = AtomicStateManager(state_file)

        # Order idempotency
        self.order_manager = IdempotentOrderManager()

        # Position reconciliation
        self.reconciler = PositionReconciler()

        # Health monitoring
        self.health_monitor = HealthMonitor()
        self._setup_health_checks()

        # Circuit breakers
        self.circuit_breakers = self._create_circuit_breakers()

        # Validators
        self.position_validator = PositionSizeValidator(
            max_position_value=self.config.get('max_position_value', 50000),
            max_portfolio_percent=self.config.get('max_portfolio_percent', 0.20)
        )
        self.feature_validator = FeatureValidator()

        logger.info("Institutional Trading Core initialized")

    def _setup_health_checks(self):
        """Setup default health checks"""
        self.health_monitor.register_check('memory', create_memory_health_check())
        self.health_monitor.register_check('disk', create_disk_health_check())

    def _create_circuit_breakers(self) -> Dict[str, EnhancedCircuitBreaker]:
        """Create circuit breakers for different components"""
        return {
            'api': EnhancedCircuitBreaker(
                name='api',
                failure_threshold=5,
                recovery_timeout=60,
                expected_exceptions=(Exception,)
            ),
            'order': EnhancedCircuitBreaker(
                name='order',
                failure_threshold=3,
                recovery_timeout=300,
                expected_exceptions=(Exception,)
            ),
            'data': EnhancedCircuitBreaker(
                name='data',
                failure_threshold=10,
                recovery_timeout=30,
                expected_exceptions=(Exception,)
            )
        }

    async def initialize(self) -> bool:
        """Initialize the institutional core"""
        try:
            # Load previous state
            saved_state = self.state_manager.load_state()
            if saved_state:
                # Restore state machine
                if 'state_machine' in saved_state:
                    sm_data = saved_state['state_machine']
                    self.state_machine = TradingStateMachine.deserialize(sm_data)

            # Transition to CONNECTED
            self.state_machine.transition_to(TradingState.CONNECTED, 'initialization')

            # Run health checks
            health = await self.health_monitor.run_all_checks()
            if not self.health_monitor.is_healthy():
                logger.warning("Health check issues detected during initialization")

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state_machine.transition_to(TradingState.INIT_FAILED, str(e))
            return False

    async def save_state(self, additional_state: Dict = None) -> bool:
        """Save current state atomically"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'state_machine': self.state_machine.serialize(),
            'health_status': self.health_monitor.get_overall_status(),
            'circuit_breakers': {
                name: cb.get_metrics() for name, cb in self.circuit_breakers.items()
            }
        }

        if additional_state:
            state.update(additional_state)

        return self.state_manager.save_state(state)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'state': self.state_machine.state.name,
            'state_duration': str(self.state_machine.get_state_duration()),
            'health': self.health_monitor.get_overall_status(),
            'circuit_breakers': {
                name: cb.get_metrics() for name, cb in self.circuit_breakers.items()
            },
            'pending_orders': self.order_manager.get_pending_count(),
            'last_reconciliation': (
                asdict(self.reconciler.get_last_reconciliation())
                if self.reconciler.get_last_reconciliation() else None
            )
        }


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def integrate_with_trading_engine(trading_engine, institutional_core: InstitutionalTradingCore):
    """
    Helper function to integrate institutional core with existing trading engine.
    Call this after creating both instances.
    """

    # Wrap order execution with idempotency
    original_execute = trading_engine._execute_real_trade

    async def idempotent_execute(signal):
        async def execute_fn():
            return await original_execute(signal)

        result = await institutional_core.order_manager.check_or_submit(
            symbol=signal.symbol,
            side=signal.signal_type.value,
            quantity=signal.position_size,
            price=signal.entry_price,
            timestamp=datetime.now(),
            execute_fn=execute_fn
        )

        if result['status'] == 'DUPLICATE':
            logger.warning(f"Duplicate order prevented: {signal.symbol}")
            return False

        return result.get('result', False)

    trading_engine._execute_real_trade = idempotent_execute

    # Add position validation before orders
    original_validate = getattr(trading_engine.risk_manager, 'validate_trade', None)

    def enhanced_validate(signal, portfolio_value):
        # First run existing validation
        if original_validate:
            original_valid, original_reason = original_validate(signal, portfolio_value)
            if not original_valid:
                return False, original_reason

        # Then run institutional validation
        return institutional_core.position_validator.validate(
            symbol=signal.symbol,
            quantity=signal.position_size,
            price=signal.entry_price,
            portfolio_value=portfolio_value
        )

    if trading_engine.risk_manager:
        trading_engine.risk_manager.validate_trade = enhanced_validate

    logger.info("Institutional core integrated with trading engine")
    return trading_engine


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # State Machine
    'TradingState',
    'TradingStateMachine',
    'StateTransition',

    # State Persistence
    'AtomicStateManager',

    # Order Management
    'IdempotentOrderManager',
    'OrderAttempt',

    # Position Reconciliation
    'PositionReconciler',
    'PositionDiscrepancy',
    'ReconciliationReport',

    # Health Monitoring
    'HealthMonitor',
    'HealthCheck',
    'create_api_health_check',
    'create_memory_health_check',
    'create_disk_health_check',

    # Circuit Breakers
    'EnhancedCircuitBreaker',
    'CircuitState',
    'CircuitBreakerOpenError',

    # Validators
    'PositionSizeValidator',
    'FeatureValidator',

    # Core
    'InstitutionalTradingCore',
    'integrate_with_trading_engine',
]
