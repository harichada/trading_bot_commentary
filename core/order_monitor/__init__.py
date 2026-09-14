"""v-phase-b-order-monitor-2026-09-14: Extract order_monitor cluster from engine.py.

Phase B of the engine modularization. The order_monitor module handles:
  - Realtime WORKING bracket/OCO order monitoring
  - Stop/TP fill detection and position sync
  - Partial fill handling with re-bracketing
  - Trailing stop replacement
  - Broker-flat detection (external close handling)
  - Orphan bracket bootstrap at startup

Public interface:
  OrderMonitor - the main monitor class, composed by TradingEngineWithCommentary

Architectural notes:
  - OrderMonitor receives a reference to the engine at construction
  - All methods that previously lived on the engine are now on OrderMonitor
  - Engine keeps thin delegate methods with same names and signatures for
    backward compatibility with the Phase A façade inventory
  - HANDS_OFF_DENYLIST, is_long_term, is_external, is_manually_managed guards
    are preserved exactly as they were
"""
from __future__ import annotations

from core.order_monitor.monitor import OrderMonitor

__all__ = ["OrderMonitor"]
