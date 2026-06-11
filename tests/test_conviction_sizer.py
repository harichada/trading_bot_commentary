"""Tests for sizing/conviction_sizer.py (shadow, operator directive
2026-06-11). Evidence basis in the module docstring: meta_proba
gradient PF 0.44 → 3.01 → 5.02 across 89 historical trades."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = REPO_ROOT / "core" / "engine.py"


class TestConvictionScore:
    def test_full_confluence_maxes_out(self):
        from sizing.conviction_sizer import conviction_score, MULT_MAX
        r = conviction_score(meta_proba=0.85, ml_agrees=True,
                             allocator_allows=True, direction=8.0,
                             volume_ratio=2.5)
        assert r.multiplier == MULT_MAX

    def test_weak_signal_floors(self):
        from sizing.conviction_sizer import conviction_score, MULT_MIN
        r = conviction_score(meta_proba=0.30, ml_agrees=False,
                             allocator_allows=False, direction=0.5,
                             volume_ratio=0.6)
        assert r.multiplier == MULT_MIN

    def test_all_missing_is_neutral(self):
        """No inputs → score 0.5 → mid-range multiplier, never an
        extreme. Absence of evidence is not evidence of weakness."""
        from sizing.conviction_sizer import conviction_score
        r = conviction_score()
        assert 0.45 <= r.score <= 0.55
        assert 0.8 <= r.multiplier <= 1.2

    def test_meta_proba_dominates(self):
        """meta_proba is the only measured predictor — moving it must
        move the multiplier more than any other single input."""
        from sizing.conviction_sizer import conviction_score
        base = conviction_score(meta_proba=0.5)
        hi_meta = conviction_score(meta_proba=0.9)
        hi_vol = conviction_score(meta_proba=0.5, volume_ratio=2.5)
        assert (hi_meta.multiplier - base.multiplier) > (
            hi_vol.multiplier - base.multiplier)

    def test_score_monotonic_in_proba(self):
        from sizing.conviction_sizer import conviction_score
        ms = [conviction_score(meta_proba=p).multiplier
              for p in (0.3, 0.5, 0.65, 0.8)]
        assert ms == sorted(ms)

    def test_read_is_frozen(self):
        import pytest
        from sizing.conviction_sizer import conviction_score
        r = conviction_score(meta_proba=0.6)
        with pytest.raises(Exception):
            r.score = 1.0  # type: ignore[misc]


class TestShadowLedger:
    def test_evaluate_writes_ndjson(self, tmp_path):
        from sizing.conviction_sizer import ConvictionSizerShadow
        ledger = tmp_path / "conv.ndjson"
        shadow = ConvictionSizerShadow(ledger_path=ledger)
        r = shadow.evaluate("mean_reversion", "NVDA",
                            meta_proba=0.72, allocator_allows=True)
        assert r is not None and r.multiplier > 1.0
        row = json.loads(ledger.read_text().splitlines()[0])
        assert row["symbol"] == "NVDA"
        assert row["would_be_multiplier"] == r.multiplier
        assert "in_meta" in row

    def test_bad_inputs_never_raise(self, tmp_path):
        from sizing.conviction_sizer import ConvictionSizerShadow
        shadow = ConvictionSizerShadow(ledger_path=tmp_path / "c.ndjson")
        r = shadow.evaluate("x", "Y", meta_proba="garbage")  # type: ignore
        assert r is None  # swallowed, logged at debug


class TestEngineWiring:
    def test_signal_router_has_conviction_hook(self):
        src = ENGINE_PATH.read_text()
        assert "_conviction_sizer" in src
        anchor = src.find("regime_allocator shadow error")
        assert anchor != -1
        window = src[anchor: anchor + 3000]
        assert "conviction" in window, (
            "conviction shadow hook must follow the allocator hook in "
            "the signal router"
        )
