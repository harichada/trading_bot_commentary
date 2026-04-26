"""Trading-cost model expressed in units of risk (R).

Converts venue fees, slippage, and market impact into a fraction of the
per-trade stop distance — the López de Prado "1R" convention (AFML §13.4).
The PnL-weighted training objective (``ml/pnl_objective.py``) consumes these
values to down-weight samples whose expected gross edge is dominated by
frictional costs; the evaluation harness (``ml/evaluation.py``) uses them to
compute cost-drag as a percentage of gross edge.

Round-trip accounting: every component is doubled because both entry and exit
incur fees and slippage. Market impact uses the Almgren-Chriss
square-root-of-participation approximation.

Defaults (``fee_bps=5, slip_bps=10, impact_coef=0.1``) match PROMPT_PACK P2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Clamp barrier_pct to at least 10 bps. A tighter stop than that is almost
# always a data artefact (zero/near-zero ATR) and would blow cost_in_r up to
# infinity. The choice is conservative: a real 10-bp stop is still expressed.
_BARRIER_PCT_FLOOR = 0.001  # 0.1% == 10 bps
_BPS_PER_UNIT = 10_000.0


@dataclass(frozen=True)
class CostModel:
    """Frozen cost parameters. All bps values are one-sided; round-trip is ×2.

    Parameters
    ----------
    fee_bps:
        Per-side venue/broker fee in basis points.
    slip_bps:
        Per-side expected slippage in basis points (quoted-spread crossing).
    impact_coef:
        Almgren-Chriss impact coefficient. The per-side impact expressed in
        basis points is ``impact_coef * sqrt(participation) * 100``. With the
        plan default (``impact_coef=0.1``) and a typical retail participation
        of 1%, this gives 1 bp per side — consistent with published buy-side
        impact calibrations. A higher coef (e.g. 0.3) lets the caller model
        larger/slower orders in less liquid names.
    """

    fee_bps: float
    slip_bps: float
    impact_coef: float

    def __post_init__(self) -> None:
        if self.fee_bps < 0 or self.slip_bps < 0 or self.impact_coef < 0:
            raise ValueError(
                "fee_bps, slip_bps, impact_coef must be non-negative "
                f"(got fee={self.fee_bps}, slip={self.slip_bps}, impact={self.impact_coef})"
            )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CostModel":
        """Load ``costs:`` block from a YAML config, falling back to defaults.

        Defaults match PROMPT_PACK P2 (fee 5bps, slip 10bps, impact 0.1).
        Silently returning defaults when the block is missing is intentional
        for this settings-style file — fail-loud kicks in only for malformed
        block content (wrong key name, wrong type).
        """
        import yaml

        data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        block = data.get("costs") or {}
        return cls(
            fee_bps=float(block.get("fee_bps", 5.0)),
            slip_bps=float(block.get("slip_bps", 10.0)),
            impact_coef=float(block.get("impact_coef", 0.1)),
        )


def _barrier_pct(
    entry_price: np.ndarray,
    atr: np.ndarray,
    sl_mult: float,
) -> np.ndarray:
    """Fractional stop distance (1R), floored to prevent division blow-ups."""
    raw = sl_mult * atr / entry_price
    return np.maximum(raw, _BARRIER_PCT_FLOOR)


def cost_in_r(
    *,
    entry_price: np.ndarray,
    atr: np.ndarray,
    sl_mult: float,
    participation: np.ndarray,
    cost_model: CostModel,
) -> np.ndarray:
    """Round-trip trading cost expressed as a fraction of 1R.

    Round-trip cost (decimal) =
        2 * (fee_bps + slip_bps) / 10_000
      + 2 * impact_coef * sqrt(participation)

    cost_in_r = round_trip_cost / barrier_pct
    where barrier_pct = sl_mult * ATR / entry_price (floored at 0.1%).

    Returns
    -------
    np.ndarray of shape (N,) in units of R.

    Raises
    ------
    ValueError
        If any input contains negative entries.
    """
    entry_price = np.asarray(entry_price, dtype="float64")
    atr = np.asarray(atr, dtype="float64")
    participation = np.asarray(participation, dtype="float64")

    if np.any(entry_price <= 0) or np.any(atr < 0) or np.any(participation < 0):
        raise ValueError(
            "entry_price must be positive; atr and participation must be non-negative"
        )

    fee_slip_component = 2.0 * (cost_model.fee_bps + cost_model.slip_bps) / _BPS_PER_UNIT
    # Impact is quoted in bps so it stacks cleanly with fees/slippage (both
    # in bps). See CostModel.impact_coef docstring for the units convention.
    impact_bps_per_side = cost_model.impact_coef * np.sqrt(participation) * 100.0
    impact_component = 2.0 * impact_bps_per_side / _BPS_PER_UNIT
    round_trip_cost = fee_slip_component + impact_component

    return round_trip_cost / _barrier_pct(entry_price, atr, sl_mult)


def ret_to_r_multiple(
    *,
    ret: np.ndarray,
    entry_price: np.ndarray,
    atr: np.ndarray,
    sl_mult: float,
) -> np.ndarray:
    """Convert decimal returns to R-multiples using the stop distance.

    1R = sl_mult * ATR / entry_price. A +1% return with a 1% stop = +1R.
    Sign is preserved.
    """
    ret = np.asarray(ret, dtype="float64")
    entry_price = np.asarray(entry_price, dtype="float64")
    atr = np.asarray(atr, dtype="float64")
    return ret / _barrier_pct(entry_price, atr, sl_mult)
