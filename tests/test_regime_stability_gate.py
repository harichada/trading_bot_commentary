"""Unit tests for RegimeStabilityGate (ml/regime.py).

The gate is a pure debouncer: maintains a "committed regime" and only
switches to a new regime once the candidate has been the classifier's
output for N consecutive events. Returns the committed regime as the
event's label — that's the regime config the bot would have actually
operated under if the gate were live.

N=5 default. Configurable.
"""
from __future__ import annotations

import pytest


def _make_gate(n: int = 5):
    from ml.regime import RegimeStabilityGate
    return RegimeStabilityGate(n=n)


class TestRegimeStabilityGate:

    def test_first_step_commits_initial_regime(self) -> None:
        g = _make_gate(n=5)
        out = g.step("trend_up_low_vol")
        assert out == "trend_up_low_vol"
        assert g.committed_regime == "trend_up_low_vol"
        assert g.pending_streak == 0

    def test_holds_committed_when_streak_below_n(self) -> None:
        """N=5; 4 consecutive `chop` after `trend_up_low_vol` → gate
        returns trend_up_low_vol for all 4."""
        g = _make_gate(n=5)
        g.step("trend_up_low_vol")  # commit
        for _ in range(4):
            assert g.step("chop") == "trend_up_low_vol"
        assert g.committed_regime == "trend_up_low_vol"

    def test_promotes_at_n(self) -> None:
        """N=5; on the 5th consecutive chop, gate switches."""
        g = _make_gate(n=5)
        g.step("trend_up_low_vol")
        for _ in range(4):
            g.step("chop")
        # 5th chop event triggers promotion.
        out = g.step("chop")
        assert out == "chop"
        assert g.committed_regime == "chop"

    def test_resets_streak_on_different_pending(self) -> None:
        """Sequence: T,T,C,C,T,R,R,R,R,R → after 5 R's, gate promotes
        to range_tight. Earlier C's and T's don't accumulate."""
        g = _make_gate(n=5)
        g.step("trend_up_low_vol")          # commit T
        g.step("trend_up_low_vol")          # holds T
        g.step("chop")                      # streak C=1
        g.step("chop")                      # streak C=2
        g.step("trend_up_low_vol")          # raw == committed → reset streak
        for i in range(5):
            label = g.step("range_tight")
            if i < 4:
                assert label == "trend_up_low_vol"  # buffer not full
            else:
                assert label == "range_tight"       # 5th promotes
        assert g.committed_regime == "range_tight"

    def test_no_change_when_raw_equals_committed(self) -> None:
        g = _make_gate(n=5)
        g.step("trend_up_low_vol")
        g.step("trend_up_low_vol")
        assert g.pending_streak == 0
        assert g.pending_regime is None

    def test_n_one_promotes_immediately(self) -> None:
        """N=1 disables debouncing — every event uses raw regime."""
        g = _make_gate(n=1)
        assert g.step("trend_up_low_vol") == "trend_up_low_vol"
        assert g.step("chop") == "chop"
        assert g.step("range_wide") == "range_wide"

    def test_n_zero_raises(self) -> None:
        from ml.regime import RegimeStabilityGate
        with pytest.raises(ValueError, match="n.*>=.*1"):
            RegimeStabilityGate(n=0)

    def test_n_negative_raises(self) -> None:
        from ml.regime import RegimeStabilityGate
        with pytest.raises(ValueError, match="n.*>=.*1"):
            RegimeStabilityGate(n=-3)

    def test_label_returned_is_committed_not_raw(self) -> None:
        """Returned label is the committed regime at that event, not the
        raw classifier output. Critical for correctly attributing trades
        to the regime config we'd actually have operated under."""
        g = _make_gate(n=5)
        g.step("trend_up_low_vol")
        # 3 raw-chops in a row — none reach the 5-event buffer, so all
        # should be labeled with the still-committed trend regime.
        labels = [g.step("chop") for _ in range(3)]
        assert all(l == "trend_up_low_vol" for l in labels), labels

    def test_unknown_regime_string_passes_through(self) -> None:
        """Gate is regime-string-agnostic — any hashable string works.
        Validation of state set is the classifier's job, not the gate's."""
        g = _make_gate(n=5)
        out = g.step("any_arbitrary_state")
        assert out == "any_arbitrary_state"

    def test_transition_count_tracked(self) -> None:
        g = _make_gate(n=2)
        g.step("trend_up_low_vol")          # initial commit (counts as 1)
        for _ in range(2):
            g.step("chop")                   # promotes on 2nd
        for _ in range(2):
            g.step("range_wide")             # promotes
        assert g.transition_count == 3       # initial + 2 promotions
