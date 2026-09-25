"""All candidate families keyed by name."""
from __future__ import annotations

from strategies.families.common import Family
from strategies.families.inplay import FAMILIES_INPLAY
from strategies.families.intraday_momentum import FAMILIES_INTRADAY_MOMENTUM
from strategies.families.intraday_reversion import FAMILIES_INTRADAY_REVERSION
from strategies.families.news import FAMILIES_NEWS
from strategies.families.swing import FAMILIES_SWING

FAMILIES: dict[str, Family] = {
    f.name: f for f in (*FAMILIES_INTRADAY_MOMENTUM, *FAMILIES_INTRADAY_REVERSION, *FAMILIES_NEWS, *FAMILIES_SWING, *FAMILIES_INPLAY)
}


def family_names() -> list[str]:
    return sorted(FAMILIES)


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown family {name!r}; valid: {family_names()}") from None
