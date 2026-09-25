"""The `FitRow` value type of the strategy-fit snapshot, and its tolerant parser.

Split out of `core/strategy_fit.py` (v-playbook-desk-b1-2026-09-25) so the reader
module stays small; `core.strategy_fit` re-exports `FitRow`, which remains the
public import path.

A row is whatever the research pipeline wrote plus defaults for anything missing —
parsing a snapshot must never raise on the live trading path, so every field is
coerced defensively and a bad value degrades to nan/0/""/False.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Dict

POOLED_SOURCE = "pooled"


@dataclass(frozen=True)
class FitRow:
    """One fit verdict from the snapshot.

    Unconditional rows are keyed (symbol, strategy, side) and carry `regime == ""`.
    Regime-conditioned rows carry the regime they were measured in; a pooled row
    (`symbol == "*"`, `source == "pooled"`) states the family-level edge for that
    regime across the whole universe and is flagged `pooled`.

    Frozen: a row is a point-in-time read from the snapshot and should never be
    mutated by a caller.
    """
    symbol: str
    strategy: str
    side: str
    verdict: str
    exp_r: float
    exp_r_lb: float
    t_stat: float
    pf: float
    wr: float
    stability: float
    decay: float
    n: int
    score: float
    source: str
    params: Dict = field(default_factory=dict)
    computed_at: str = ""
    regime: str = ""
    pooled: bool = False

    @staticmethod
    def from_dict(d: Dict) -> "FitRow":
        """Build a FitRow from a raw snapshot row dict.

        Tolerant of missing keys (default to nan/0/""/False) and of a
        `params_json` field that is a JSON string rather than a dict.
        """
        def _f(key: str, default: float) -> float:
            try:
                return float(d.get(key, default))
            except (TypeError, ValueError):
                return default

        def _i(key: str, default: int) -> int:
            try:
                return int(d.get(key, default))
            except (TypeError, ValueError):
                return default

        def _s(key: str, default: str) -> str:
            v = d.get(key, default)
            return v if isinstance(v, str) else default

        params = d.get("params")
        if not isinstance(params, dict):
            raw = d.get("params_json")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else {}
                params = parsed if isinstance(parsed, dict) else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                params = {}

        source = _s("source", "")
        return FitRow(
            symbol=_s("symbol", ""),
            strategy=_s("strategy", ""),
            side=_s("side", ""),
            verdict=_s("verdict", ""),
            exp_r=_f("exp_r", float("nan")),
            exp_r_lb=_f("exp_r_lb", float("nan")),
            t_stat=_f("t_stat", float("nan")),
            pf=_f("pf", float("nan")),
            wr=_f("wr", float("nan")),
            stability=_f("stability", float("nan")),
            decay=_f("decay", float("nan")),
            n=_i("n", 0),
            score=_f("score", float("nan")),
            source=source,
            params=params,
            computed_at=_s("computed_at", ""),
            regime=_s("regime", ""),
            # The writer marks a pooled lane by its source; an explicit flag is
            # honoured too so a hand-written snapshot can say so directly.
            pooled=bool(d.get("pooled", False)) or source == POOLED_SOURCE,
        )

    def to_json_dict(self) -> Dict:
        """JSON-safe dict of this row: every non-finite float becomes None.

        v-strategy-fit-review2-2026-09-25 (Important 5): cold-start rows
        legitimately carry NaN metrics (`from_dict` defaults every missing
        numeric to nan). `json.dumps` emits bare `NaN`, which is not valid
        JSON — FastAPI's encoder rejects it and `GET /api/strategy-fit`
        answered 500 for any symbol with a cold-start row. `None` is the
        honest wire representation of "not measured".
        """
        out: Dict = {}
        for key, value in dataclasses.asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
            else:
                out[key] = value
        return out
