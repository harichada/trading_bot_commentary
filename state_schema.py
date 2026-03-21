"""Pydantic models for trader state validation.

Validates state before persistence to catch corruption early.
Used by _save_state() in gap_fade_app.py.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


class PositionSchema(BaseModel):
    symbol: str
    shares: int
    entry_price: float
    stop_price: float
    half_target: float
    full_target: float
    prev_close: float
    entry_time: str
    remaining_shares: int
    entry_fill_price: float = 0.0
    entry_order_id: str = ''
    stop_order_id: str = ''
    direction: str = 'short'
    gap_pct: float = 0.0
    vol_ratio: float = 0.0
    score: float = 0.0
    catalyst: str = ''
    close_on_open: bool = False
    closing: bool = False
    high_water_pnl: float = 0.0
    price_history: List[float] = []
    llm_hold_overrides: int = 0
    source: str = 'gap_fade'
    strategy_id: str = ''

    @field_validator('shares', 'remaining_shares')
    @classmethod
    def positive_shares(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f'shares must be >= 0, got {v}')
        return v

    @field_validator('entry_price', 'stop_price')
    @classmethod
    def positive_price(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f'price must be >= 0, got {v}')
        return v

    @field_validator('direction')
    @classmethod
    def valid_direction(cls, v: str) -> str:
        if v not in ('short', 'long'):
            raise ValueError(f"direction must be 'short' or 'long', got '{v}'")
        return v

    class Config:
        extra = 'allow'


class DailyStatsSchema(BaseModel):
    date: str = ''
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ''

    class Config:
        extra = 'allow'


class TraderStateSchema(BaseModel):
    status: str = 'stopped'
    equity: float = 0.0
    peak_equity: float = 0.0
    positions: Dict[str, Any] = {}
    daily_stats: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
    saved_at: str = ''
    last_updated: str = ''

    @field_validator('equity')
    @classmethod
    def non_negative_equity(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f'equity must be >= 0, got {v}')
        return v

    @field_validator('positions')
    @classmethod
    def validate_positions(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        for sym, pos_data in v.items():
            try:
                PositionSchema(**pos_data)
            except Exception as e:
                errors.append(f'{sym}: {e}')
        if errors:
            raise ValueError(f'Invalid positions: {"; ".join(errors)}')
        return v

    class Config:
        extra = 'allow'
