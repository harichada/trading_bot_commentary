import pandas as pd
import pytest


def test_entry_validation_and_risk():
    from strategies.families.models import Entry
    e = Entry("NVDA", "long", pd.Timestamp("2025-01-02 09:35"), 100.0, 98.0, 104.0, 12, "intraday")
    assert e.risk_per_share == pytest.approx(2.0)
    s = Entry("NVDA", "short", pd.Timestamp("2025-01-02 09:35"), 100.0, 101.5, 97.0, 12, "intraday")
    assert s.risk_per_share == pytest.approx(1.5)
    with pytest.raises(ValueError):
        Entry("NVDA", "buy", pd.Timestamp("2025-01-02"), 1, 0.9, 1.2, 1, "intraday")
    with pytest.raises(ValueError):
        Entry("NVDA", "long", pd.Timestamp("2025-01-02"), 1, 0.9, 1.2, 1, "weekly")
    with pytest.raises(ValueError):
        Entry("NVDA", "long", pd.Timestamp("2025-01-02"), 1, 1.1, 1.2, 1, "intraday")   # stop above entry
    with pytest.raises(ValueError):
        Entry("NVDA", "short", pd.Timestamp("2025-01-02"), 1, 1.1, 1.05, 1, "intraday")  # target above entry
    with pytest.raises(Exception):
        e.entry_px = 2.0  # type: ignore[misc]


def test_overnight_is_a_valid_hold_kind():
    """Batch-1 Task 1: the overnight hold kind must survive the Entry contract (and therefore the
    fit ledger, the shadow resolver and the live signal adapter, none of which whitelist it)."""
    from strategies.families.models import HOLD_KINDS, Entry
    assert HOLD_KINDS == ("intraday", "multiday", "overnight")
    e = Entry("NVDA", "long", pd.Timestamp("2025-01-02 15:55"), 100.0, 97.0, 160.0, 1, "overnight")
    assert e.hold_kind == "overnight" and e.risk_per_share == pytest.approx(3.0)
