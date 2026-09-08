"""v-conviction-floor-065: regression test.

2026-09-02: both live losers (NVAX meta 0.6235, HYMC meta 0.6117)
cleared the 0.60 conviction floor. Counterfactual over all 35
mean-rev-long trades carrying meta_proba (5/8–9/2): the 0.60–0.65
band held 9 trades netting -$247.27; raising the floor to 0.65 turns
the strategy from PF 0.98 (-$45.79) to PF 3.70 (+$1,208.13) on the
kept set. Kill-switch unchanged: trading.enable_conviction_floor_meanrev.
"""

from core.config import Config


def test_conviction_floor_default_is_065():
    assert Config().CONVICTION_FLOOR_META == 0.65


def test_conviction_floor_kill_switch_still_present():
    assert isinstance(Config().ENABLE_CONVICTION_FLOOR_MEANREV, bool)
