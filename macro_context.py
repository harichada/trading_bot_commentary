"""
FOMC meeting dates and macro context utilities for backtesting.

Provides date-level checks for whether a trading day falls on or near
an FOMC meeting, which historically affects gap-fade profitability.
"""

from datetime import date, datetime, timedelta
from typing import FrozenSet

# ── FOMC Meeting Dates (both days of each two-day meeting) ──────────────
# Source: federalreserve.gov scheduled FOMC meetings

FOMC_DATES: FrozenSet[str] = frozenset([
    # 2024
    "2024-01-30", "2024-01-31",
    "2024-03-19", "2024-03-20",
    "2024-04-30", "2024-05-01",
    "2024-06-11", "2024-06-12",
    "2024-07-30", "2024-07-31",
    "2024-09-17", "2024-09-18",
    "2024-11-06", "2024-11-07",
    "2024-12-17", "2024-12-18",
    # 2025
    "2025-01-28", "2025-01-29",
    "2025-03-18", "2025-03-19",
    "2025-05-06", "2025-05-07",
    "2025-06-17", "2025-06-18",
    "2025-07-29", "2025-07-30",
    "2025-09-16", "2025-09-17",
    "2025-10-28", "2025-10-29",
    "2025-12-09", "2025-12-10",
    # 2026
    "2026-01-27", "2026-01-28",
    "2026-03-17", "2026-03-18",
    "2026-04-28", "2026-04-29",
    "2026-06-16", "2026-06-17",
    "2026-07-28", "2026-07-29",
    "2026-09-15", "2026-09-16",
    "2026-10-27", "2026-10-28",
    "2026-12-08", "2026-12-09",
])


def _parse_date(date_str: str) -> date:
    """Parse YYYY-MM-DD string to date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def is_fomc_day(date_str: str) -> bool:
    """Check if a date is an FOMC meeting day.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.

    Returns:
        True if the date is an FOMC meeting day.
    """
    return date_str in FOMC_DATES


def is_fomc_week(date_str: str) -> bool:
    """Check if a date falls within an FOMC meeting week (Mon-Fri).

    The FOMC week is defined as the ISO week (Monday through Sunday)
    containing any FOMC meeting day.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.

    Returns:
        True if the date is in the same ISO week as an FOMC meeting.
    """
    target = _parse_date(date_str)
    target_year, target_week, _ = target.isocalendar()

    for fomc_str in FOMC_DATES:
        fomc_date = _parse_date(fomc_str)
        fomc_year, fomc_week, _ = fomc_date.isocalendar()
        if target_year == fomc_year and target_week == fomc_week:
            return True

    return False


if __name__ == "__main__":
    # Quick self-test
    assert is_fomc_day("2024-01-31") is True
    assert is_fomc_day("2024-01-29") is False
    assert is_fomc_week("2024-01-29") is True   # Mon of FOMC week
    assert is_fomc_week("2024-02-02") is True    # Fri of same FOMC week
    assert is_fomc_week("2024-02-05") is False   # Next week
    print(f"FOMC dates loaded: {len(FOMC_DATES)} days across 24 meetings")
    print("All self-tests passed.")
