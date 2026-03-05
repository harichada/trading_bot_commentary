"""Multi-broker coordinator — runs Alpaca + OANDA (or more) in parallel.

Manages independent GapFadeLiveTrader instances (one per broker), provides
combined state/risk views, and proxies attribute access to the primary
trader for backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # GapFadeLiveTrader imported lazily to avoid circular deps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BrokerSlot — one entry per broker
# ---------------------------------------------------------------------------

@dataclass
class BrokerSlot:
    """Registered broker within the coordinator."""
    broker_id: str
    trader: Any  # GapFadeLiveTrader (quoted to avoid circular import)
    label: str = ''
    enabled: bool = True


# ---------------------------------------------------------------------------
# CombinedRiskManager — cross-broker risk limits
# ---------------------------------------------------------------------------

@dataclass
class CombinedRiskManager:
    """Aggregate risk checks across all active brokers.

    All thresholds are fractions (e.g. 0.03 = 3%).
    """
    combined_daily_loss_limit: float = 0.03   # halt all if combined daily loss > 3%
    combined_max_drawdown: float = 0.07       # halt all if combined drawdown > 7%
    max_total_positions: int = 8              # max positions across all brokers

    def check_combined_risk(self, slots: Dict[str, BrokerSlot]) -> Dict[str, Any]:
        """Check cross-broker risk limits.

        Returns dict of violation keys → details.  Empty dict = all clear.
        """
        violations: Dict[str, Any] = {}

        total_equity = 0.0
        total_peak = 0.0
        total_daily_pnl = 0.0
        total_positions = 0

        for slot in slots.values():
            if not slot.enabled:
                continue
            eng = slot.trader.engine
            total_equity += eng.equity
            total_peak += eng.peak_equity
            total_daily_pnl += eng.daily_stats.pnl
            total_positions += len(eng.positions)

        # Daily loss check
        if total_peak > 0 and total_daily_pnl < 0:
            daily_loss_pct = abs(total_daily_pnl) / total_peak
            if daily_loss_pct >= self.combined_daily_loss_limit:
                violations['daily_loss'] = {
                    'pct': daily_loss_pct,
                    'limit': self.combined_daily_loss_limit,
                    'pnl': total_daily_pnl,
                }

        # Drawdown check
        if total_peak > 0:
            dd = (total_peak - total_equity) / total_peak
            if dd >= self.combined_max_drawdown:
                violations['drawdown'] = {
                    'pct': dd,
                    'limit': self.combined_max_drawdown,
                    'equity': total_equity,
                    'peak': total_peak,
                }

        # Position count check
        if total_positions > self.max_total_positions:
            violations['position_count'] = {
                'count': total_positions,
                'limit': self.max_total_positions,
            }

        return violations

    def get_status(self, slots: Dict[str, BrokerSlot]) -> Dict[str, Any]:
        """Aggregate status across all brokers."""
        total_equity = 0.0
        total_peak = 0.0
        total_daily_pnl = 0.0
        total_positions = 0

        for slot in slots.values():
            if not slot.enabled:
                continue
            eng = slot.trader.engine
            total_equity += eng.equity
            total_peak += eng.peak_equity
            total_daily_pnl += eng.daily_stats.pnl
            total_positions += len(eng.positions)

        violations = self.check_combined_risk(slots)
        return {
            'total_equity': total_equity,
            'total_peak': total_peak,
            'total_daily_pnl': total_daily_pnl,
            'total_positions': total_positions,
            'violations': violations,
            'halted': bool(violations.get('daily_loss') or violations.get('drawdown')),
        }


# ---------------------------------------------------------------------------
# MultiBrokerCoordinator
# ---------------------------------------------------------------------------

class MultiBrokerCoordinator:
    """Manages multiple GapFadeLiveTrader instances, one per broker.

    Attributes on this object proxy to the *primary* trader (first added)
    for backward compatibility with code that uses ``live_trader.xxx``.
    """

    def __init__(self):
        self._slots: Dict[str, BrokerSlot] = {}
        self._primary_id: Optional[str] = None
        self._combined_risk = CombinedRiskManager()
        self._llm_supervisor = None  # shared LLM, set by setup_shared_llm()

    # -- Registration --------------------------------------------------------

    def add_broker(
        self,
        broker_id: str,
        trader: Any,
        label: str = '',
        enabled: bool = True,
    ) -> None:
        """Register a broker/trader pair.  First added becomes primary."""
        self._slots[broker_id] = BrokerSlot(
            broker_id=broker_id,
            trader=trader,
            label=label or broker_id,
            enabled=enabled,
        )
        # Wire up coordinator back-reference
        trader._coordinator = self
        trader._combined_risk_checker = self._make_risk_checker()
        # Propagate to engine so should_enter() can check combined risk
        if hasattr(trader, 'engine') and trader.engine is not None:
            trader.engine._combined_risk_checker = trader._combined_risk_checker

        if self._primary_id is None:
            self._primary_id = broker_id

    def _make_risk_checker(self) -> Callable:
        """Return a callable that checks combined risk for use in engine."""
        def _checker():
            return self._combined_risk.check_combined_risk(self._slots)
        return _checker

    # -- Accessors -----------------------------------------------------------

    @property
    def primary(self):
        """The primary (first-added) trader — backward compat."""
        if self._primary_id and self._primary_id in self._slots:
            return self._slots[self._primary_id].trader
        raise RuntimeError("No primary broker registered")

    def get_trader(self, broker_id: str):
        """Get a specific trader by broker ID."""
        slot = self._slots.get(broker_id)
        if slot is None:
            raise KeyError(f"No broker registered with id '{broker_id}'")
        return slot.trader

    def get_trader_for_symbol(self, symbol: str):
        """Find the trader that has a position in *symbol*, or primary."""
        for slot in self._slots.values():
            if slot.enabled and symbol in slot.trader.engine.positions:
                return slot.trader
        return self.primary

    @property
    def active_traders(self) -> List:
        """All enabled traders."""
        return [s.trader for s in self._slots.values() if s.enabled]

    @property
    def all_slots(self) -> Dict[str, BrokerSlot]:
        return dict(self._slots)

    @property
    def broker_ids(self) -> List[str]:
        return list(self._slots.keys())

    # -- Lifecycle -----------------------------------------------------------

    async def start_all(self) -> None:
        """Start all enabled traders."""
        for slot in self._slots.values():
            if slot.enabled and slot.trader.status == 'stopped':
                await slot.trader.start()

    async def stop_all(self) -> None:
        """Stop all traders."""
        for slot in self._slots.values():
            await slot.trader.stop()

    async def start_broker(self, broker_id: str) -> None:
        slot = self._slots.get(broker_id)
        if slot is None:
            raise KeyError(f"Unknown broker: {broker_id}")
        slot.enabled = True
        if slot.trader.status == 'stopped':
            await slot.trader.start()

    async def stop_broker(self, broker_id: str) -> None:
        slot = self._slots.get(broker_id)
        if slot is None:
            raise KeyError(f"Unknown broker: {broker_id}")
        await slot.trader.stop()
        slot.enabled = False

    # -- Combined State ------------------------------------------------------

    def get_combined_state(self) -> Dict[str, Any]:
        """Merged state across all brokers for API/dashboard."""
        brokers_state = {}
        total_equity = 0.0
        total_daily_pnl = 0.0
        total_positions = 0
        all_positions = {}

        for bid, slot in self._slots.items():
            eng = slot.trader.engine
            equity = eng.equity
            daily_pnl = eng.daily_stats.pnl
            positions = {s: self._pos_to_dict(p, bid)
                         for s, p in eng.positions.items()}

            brokers_state[bid] = {
                'label': slot.label,
                'enabled': slot.enabled,
                'status': slot.trader.status,
                'equity': equity,
                'daily_pnl': daily_pnl,
                'position_count': len(positions),
                'positions': positions,
            }

            if slot.enabled:
                total_equity += equity
                total_daily_pnl += daily_pnl
                total_positions += len(positions)
                all_positions.update(positions)

        risk_status = self._combined_risk.get_status(self._slots)

        return {
            'total_equity': total_equity,
            'total_daily_pnl': total_daily_pnl,
            'total_positions': total_positions,
            'all_positions': all_positions,
            'brokers': brokers_state,
            'risk': risk_status,
        }

    @staticmethod
    def _pos_to_dict(pos, broker_id: str) -> Dict:
        """Convert a GapPosition to a dict with broker_id tag."""
        from dataclasses import asdict
        d = asdict(pos)
        d['broker_id'] = broker_id
        return d

    # -- Shared LLM ----------------------------------------------------------

    def setup_shared_llm(self, config) -> None:
        """Create one LLM supervisor and share it across all traders."""
        # Import here to avoid circular imports
        from gap_fade_app import LLMSupervisor
        self._llm_supervisor = LLMSupervisor(config)

        # Append multi-broker awareness to system prompt
        if len(self._slots) > 1:
            broker_lines = []
            for slot in self._slots.values():
                mh = slot.trader._market_hours
                broker_lines.append(f"- {slot.label} ({slot.broker_id}): {type(mh).__name__}")
            addendum = (
                "\n\n## MULTI-BROKER AWARENESS\n"
                "You are monitoring MULTIPLE markets simultaneously:\n"
                + "\n".join(broker_lines) + "\n"
                "Consider combined risk (total equity, total drawdown) in decisions.\n"
                "Specify which broker actions apply to."
            )
            self._llm_supervisor._SYSTEM_PROMPT += addendum

        for slot in self._slots.values():
            slot.trader.llm_supervisor = self._llm_supervisor
            logger.info("Shared LLM assigned to broker %s", slot.broker_id)

    def build_combined_llm_state(self, now) -> Dict[str, Any]:
        """Build merged LLM state from all traders."""
        combined = {
            'multi_broker': True,
            'brokers': {},
            'total_equity': 0.0,
            'total_daily_pnl': 0.0,
            'total_positions': 0,
        }

        for bid, slot in self._slots.items():
            if not slot.enabled:
                continue
            state = slot.trader._build_llm_state(now)
            combined['brokers'][bid] = state
            combined['total_equity'] += state.get('equity', 0)
            combined['total_daily_pnl'] += state.get('daily_pnl', 0)
            combined['total_positions'] += len(state.get('positions', {}))

        # Use primary's time info
        primary_state = combined['brokers'].get(self._primary_id, {})
        combined['time_et'] = primary_state.get('time_et', '')
        combined['day_of_week'] = primary_state.get('day_of_week', '')

        return combined

    # -- Backward compatibility proxy ----------------------------------------

    # Coordinator's own attributes — must NOT proxy to avoid infinite recursion
    _COORDINATOR_ATTRS = frozenset({
        '_slots', '_primary_id', '_combined_risk', '_llm_supervisor',
        'primary', 'get_trader', 'active_traders', 'all_slots', 'broker_ids',
        'add_broker', 'start_all', 'stop_all', 'start_broker', 'stop_broker',
        'get_combined_state', 'get_trader_for_symbol',
        'build_combined_llm_state', 'setup_shared_llm',
    })

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the primary trader.

        This allows ``live_trader.status``, ``live_trader.config``, etc.
        to work unchanged when ``live_trader`` is actually a coordinator.
        """
        # Block only coordinator's own attrs to prevent infinite recursion
        if name in self._COORDINATOR_ATTRS:
            raise AttributeError(name)
        return getattr(self.primary, name)
