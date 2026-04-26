"""Tests for the PnL cost model (López de Prado AFML §13.4 — costs in R units).

The cost model converts venue fees, slippage, and market impact into a fraction
of the per-trade risk denominator (1R). This is the unit that the PnL-weighted
training objective consumes, so every test pins the math exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.costs import CostModel, cost_in_r, ret_to_r_multiple


class TestCostInR:
    def test_symmetric_barriers_known_cost(self) -> None:
        """Fee 5bps + slip 10bps round-trip → 30bps. 1R = 1% → cost_in_r ≈ 0.30."""
        model = CostModel(fee_bps=5.0, slip_bps=10.0, impact_coef=0.0)
        entry = np.array([100.0])
        atr = np.array([1.0])
        sl_mult = 1.0

        out = cost_in_r(
            entry_price=entry, atr=atr, sl_mult=sl_mult,
            participation=np.array([0.0]), cost_model=model,
        )

        assert out == pytest.approx(np.array([0.30]), abs=1e-6)

    def test_participation_zero_zero_impact(self) -> None:
        """impact_coef * sqrt(0) = 0 — impact vanishes with zero participation."""
        model = CostModel(fee_bps=5.0, slip_bps=10.0, impact_coef=0.5)
        entry = np.array([100.0])
        atr = np.array([1.0])

        out = cost_in_r(
            entry_price=entry, atr=atr, sl_mult=1.0,
            participation=np.array([0.0]), cost_model=model,
        )

        # Only fee+slip component, same as impact_coef=0.
        assert out == pytest.approx(np.array([0.30]), abs=1e-6)

    def test_impact_scales_with_sqrt_participation(self) -> None:
        """Square-root market impact (Almgren-Chriss). Units convention:
        impact_bps_per_side = impact_coef * sqrt(participation) * 100.
        """
        model = CostModel(fee_bps=0.0, slip_bps=0.0, impact_coef=0.1)
        entry = np.array([100.0, 100.0])
        atr = np.array([1.0, 1.0])
        # participation 0.04 → sqrt=0.2 → bps/side = 0.1 * 0.2 * 100 = 2.0
        # participation 0.01 → sqrt=0.1 → bps/side = 0.1 * 0.1 * 100 = 1.0
        # Round-trip = 4 bps and 2 bps. 1R = 1% = 100 bps → 0.04R and 0.02R.
        out = cost_in_r(
            entry_price=entry, atr=atr, sl_mult=1.0,
            participation=np.array([0.04, 0.01]), cost_model=model,
        )

        assert out[0] == pytest.approx(0.04, abs=1e-6)
        assert out[1] == pytest.approx(0.02, abs=1e-6)

    def test_vectorized_shape_matches_input(self) -> None:
        """Output length N for input length N — no broadcasting surprises."""
        n = 17
        model = CostModel(fee_bps=5.0, slip_bps=10.0, impact_coef=0.1)
        out = cost_in_r(
            entry_price=np.full(n, 100.0),
            atr=np.full(n, 1.0),
            sl_mult=1.0,
            participation=np.full(n, 0.01),
            cost_model=model,
        )
        assert out.shape == (n,)
        assert np.all(np.isfinite(out))

    def test_barrier_pct_floor_prevents_explosion(self) -> None:
        """When sl_mult*ATR/entry < 0.1%, clamp to 0.1% (prevents /0 blow-up)."""
        model = CostModel(fee_bps=5.0, slip_bps=10.0, impact_coef=0.0)
        # Tiny ATR → would give barrier_pct ~ 0.00001 = 0.001% → clamp
        out = cost_in_r(
            entry_price=np.array([100.0]),
            atr=np.array([0.001]),
            sl_mult=1.0,
            participation=np.array([0.0]),
            cost_model=model,
        )
        # 30bps / clamped 0.1% = 30bps / 10bps = 3.0R
        assert out[0] == pytest.approx(3.0, abs=1e-6)

    def test_negative_inputs_rejected(self) -> None:
        model = CostModel(fee_bps=5.0, slip_bps=10.0, impact_coef=0.0)
        with pytest.raises(ValueError):
            cost_in_r(
                entry_price=np.array([-1.0]),
                atr=np.array([1.0]),
                sl_mult=1.0,
                participation=np.array([0.0]),
                cost_model=model,
            )


class TestRetToRMultiple:
    def test_one_percent_move_equals_one_r_when_barrier_is_one_percent(self) -> None:
        # entry=100, ATR=1, sl_mult=1 → barrier_pct = 1%. Return 1% → 1R.
        r = ret_to_r_multiple(
            ret=np.array([0.01, -0.01, 0.02]),
            entry_price=np.array([100.0, 100.0, 100.0]),
            atr=np.array([1.0, 1.0, 1.0]),
            sl_mult=1.0,
        )
        assert r == pytest.approx(np.array([1.0, -1.0, 2.0]), abs=1e-9)

    def test_ret_to_r_preserves_sign(self) -> None:
        r = ret_to_r_multiple(
            ret=np.array([-0.005]),
            entry_price=np.array([50.0]),
            atr=np.array([0.5]),
            sl_mult=1.0,
        )
        assert r[0] < 0


class TestCostModelDataclass:
    def test_is_frozen(self) -> None:
        model = CostModel(fee_bps=5.0, slip_bps=10.0, impact_coef=0.1)
        with pytest.raises(Exception):
            model.fee_bps = 7.0  # type: ignore[misc]

    def test_from_yaml_reads_costs_block(self, tmp_path: Path) -> None:
        import yaml
        config = {"costs": {"fee_bps": 7.0, "slip_bps": 12.0, "impact_coef": 0.2}}
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(config))

        model = CostModel.from_yaml(path)

        assert model.fee_bps == 7.0
        assert model.slip_bps == 12.0
        assert model.impact_coef == 0.2

    def test_from_yaml_uses_defaults_when_block_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("other_key: value\n")

        model = CostModel.from_yaml(path)

        # Defaults per PROMPT_PACK P2 spec.
        assert model.fee_bps == 5.0
        assert model.slip_bps == 10.0
        assert model.impact_coef == 0.1

    def test_negative_fee_rejected(self) -> None:
        with pytest.raises(ValueError):
            CostModel(fee_bps=-1.0, slip_bps=10.0, impact_coef=0.1)
