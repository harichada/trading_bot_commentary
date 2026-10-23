"""Frozen params + constants for the swing lane (SPEC §1, §3, §4)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARAMS_PATH = HERE / "best_params.json"
UNIVERSE_PATH = HERE / "universe.json"
REGIME_ONLY = frozenset({"SPY", "QQQ"})

CODE_VERSION = "swing_lane-v2-2026-10-23"

# SPEC §3/§4 constants (frozen from the backtest)
MIN_PRICE = 5.0
MIN_DVOL = 20e6
MIN_BARS = 260
RISK_USD = 150.0
MAX_NOTIONAL = 15000.0
MAX_OPEN = 4
MAX_OPEN_RISK_USD = 600.0
DAILY_LOSS_LIMIT_USD = 500.0
EARNINGS_BLACKOUT_DAYS = 5
TIME_STOP_DAYS = 10
SCAN_WINDOW = ((16, 20), (16, 30))   # America/New_York, inclusive
COMPLETE_BAR_AFTER = (16, 15)

# setup letter -> (config flag property, module name or None if not implemented)
SETUPS = {
    "A": ("SWING_SETUP_PULLBACK", None),   # failed backtest: research only, no scanner shipped
    "B": ("SWING_SETUP_BREAKOUT", "breakout"),
    "C": ("SWING_SETUP_GAPDRIFT", "gapdrift"),
    "D": ("SWING_SETUP_MEANREV", None),    # failed backtest: research only, no scanner shipped
}


def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def load_universe(path: Path = UNIVERSE_PATH) -> list:
    with open(path) as f:
        return [str(s).upper() for s in json.load(f)]


@dataclass
class LaneSettings:
    """Snapshot of the relevant Config flags, read once per run."""
    shadow: bool = False
    live: bool = False
    setups: dict = field(default_factory=lambda: {k: False for k in SETUPS})
    earnings_blackout: bool = True
    denylist: frozenset = frozenset()
    data_root: str = "data/strategy_hunt"
    log_dir: str = "data/swing_lane"
    earnings_provider: str = "none"

    @classmethod
    def from_config(cls, cfg) -> "LaneSettings":
        return cls(
            shadow=bool(cfg.ENABLE_SWING_SHADOW),
            live=bool(cfg.ENABLE_SWING_LIVE),
            setups={k: bool(getattr(cfg, flag)) for k, (flag, _) in SETUPS.items()},
            earnings_blackout=bool(cfg.SWING_EARNINGS_BLACKOUT),
            denylist=frozenset(s.upper() for s in cfg.HANDS_OFF_DENYLIST),
            data_root=str(cfg.STRATEGY_HUNT_DATA_ROOT),
            earnings_provider=str(getattr(cfg, "SWING_EARNINGS_PROVIDER", "none")),
        )
