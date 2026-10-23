"""Each setup exposes scan(bars, ind, params) -> list[Signal]. Imported lazily only when enabled."""
from dataclasses import dataclass, field


@dataclass
class Signal:
    setup: str
    sym: str
    signal_date: object
    close: float
    atr14: float
    rvol: float
    gap: float
    rng20_atr: float
    planned_stop: float
    planned_target: object  # float or None (B trails)
    score: float
    extra: dict = field(default_factory=dict)
