"""Risk Console knob registry, validation and tighten/loosen classification.

Pure module: no imports of core.config, no dotenv, no side effects at import
time. Ratchet rule: every change to a live knob is classified as "tighten"
(more conservative), "loosen" (more permissive) or "noop" so callers can
gate loosening changes behind extra review. `coerce` validates and
normalises a raw value for a knob; `classify` compares an old/new pair of
already-coerced values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Knob:
    key: str
    yaml_path: str
    env_name: str
    kind: str
    group: str
    label: str
    loosen: str
    restart_required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple[str, ...] = field(default_factory=tuple)


ALLOCATION_CEILING_PATH = "trading.settings_console.allocation_ceiling"
DEFAULT_ALLOCATION_CEILING = 5000.0

KNOWN_FIT_FAMILIES = (
    "mr_20d", "mom_12_1", "dt_momentum", "gap_play", "vwap_reversion",
    "mean_rev_intraday", "low_vol_trend", "orb_inplay", "pead_smallcap", "rs_leader",
)

_ALLOWLIST_SIDES = ("long", "short")
_ALLOWLIST_REGIMES = ("risk_on", "risk_off", "mixed")

KNOBS: tuple[Knob, ...] = (
    Knob(
        key="bot_capital_allocation", yaml_path="trading.bot_capital_allocation",
        env_name="BOT_CAPITAL_ALLOCATION", kind="money", group="risk",
        label="Bot capital allocation", loosen="increase", minimum=0.0,
    ),
    Knob(
        key="risk_per_trade_pct", yaml_path="trading.risk_per_trade_pct",
        env_name="RISK_PER_TRADE_PCT", kind="fraction", group="risk",
        label="Risk per trade (%)", loosen="increase", minimum=0.0, maximum=1.0,
    ),
    Knob(
        key="max_daily_loss", yaml_path="trading.max_daily_loss",
        env_name="MAX_DAILY_LOSS", kind="fraction", group="risk",
        label="Max daily loss (%)", loosen="increase", minimum=0.0, maximum=1.0,
    ),
    Knob(
        key="max_positions", yaml_path="trading.max_positions",
        env_name="", kind="int", group="risk",
        label="Max positions", loosen="increase", minimum=1, maximum=50,
    ),
    Knob(
        key="swing_max_hold_sessions", yaml_path="trading.swing_max_hold_sessions",
        env_name="SWING_MAX_HOLD_SESSIONS", kind="int", group="risk",
        label="Swing max hold sessions", loosen="increase", minimum=1, maximum=60,
    ),
    Knob(
        key="orb_live_entries_enabled", yaml_path="trading.orb_live_entries_enabled",
        env_name="ORB_LIVE_ENTRIES_ENABLED", kind="bool", group="lanes",
        label="ORB live entries", loosen="true",
    ),
    Knob(
        key="day_trade_live_entries_enabled", yaml_path="trading.day_trade_live_entries_enabled",
        env_name="DAY_TRADE_LIVE_ENTRIES_ENABLED", kind="bool", group="lanes",
        label="Day trade live entries", loosen="true",
    ),
    Knob(
        key="day_trade_short_live_entries_enabled", yaml_path="trading.day_trade_short_live_entries_enabled",
        env_name="DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED", kind="bool", group="lanes",
        label="Day trade short live entries", loosen="true",
    ),
    Knob(
        key="mean_rev_live_entries_enabled", yaml_path="trading.mean_rev_live_entries_enabled",
        env_name="MEAN_REV_LIVE_ENTRIES_ENABLED", kind="bool", group="lanes",
        label="Mean reversion live entries", loosen="true",
    ),
    Knob(
        key="fit_lane_live_entries_enabled", yaml_path="trading.fit_lane_live_entries_enabled",
        env_name="FIT_LANE_LIVE_ENTRIES_ENABLED", kind="csv", group="lanes",
        label="Fit lane live entries", loosen="add", allowed=KNOWN_FIT_FAMILIES,
    ),
    Knob(
        key="fit_lane_live_allowlist", yaml_path="trading.fit_lane_live_allowlist",
        env_name="FIT_LANE_LIVE_ALLOWLIST", kind="csv", group="lanes",
        label="Fit lane live allowlist", loosen="add", allowed=KNOWN_FIT_FAMILIES,
    ),
    Knob(
        key="fit_lanes_enabled", yaml_path="trading.fit_lanes_enabled",
        env_name="FIT_LANES_ENABLED", kind="csv", group="lanes",
        label="Fit lanes enabled", loosen="add", allowed=KNOWN_FIT_FAMILIES,
        restart_required=True,
    ),
    Knob(
        key="enable_playbook_gate", yaml_path="trading.enable_playbook_gate",
        env_name="ENABLE_PLAYBOOK_GATE", kind="bool", group="gates",
        label="Playbook gate", loosen="false",
    ),
    Knob(
        key="enable_desk_limits", yaml_path="trading.enable_desk_limits",
        env_name="ENABLE_DESK_LIMITS", kind="bool", group="gates",
        label="Desk limits", loosen="false",
    ),
)

KNOB_BY_KEY: dict[str, Knob] = {knob.key: knob for knob in KNOBS}

_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}


def _coerce_fraction(knob: Knob, raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{knob.label} must be a number between {knob.minimum} and {knob.maximum}")
    # Fraction knobs use an open-at-zero, closed-at-max range: (minimum, maximum].
    if value != value or value <= knob.minimum or value > knob.maximum:
        raise ValueError(f"{knob.label} must be greater than {knob.minimum} and at most {knob.maximum} (got {raw!r})")
    return value


def _coerce_money(knob: Knob, raw: Any, ceiling: float | None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{knob.label} must be a positive number")
    limit = DEFAULT_ALLOCATION_CEILING if ceiling is None else ceiling
    if value <= 0:
        raise ValueError(f"{knob.label} must be greater than 0")
    if value > limit:
        raise ValueError(f"{knob.label} must not exceed the allocation ceiling of {limit}")
    return value


def _coerce_int(knob: Knob, raw: Any) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{knob.label} must be an integer between {knob.minimum} and {knob.maximum}")
    try:
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped or not stripped.lstrip("-").isdigit():
                raise ValueError
            value = int(stripped)
        else:
            value = int(raw)
            if value != raw:
                raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{knob.label} must be an integer between {knob.minimum} and {knob.maximum} (got {raw!r})")
    if knob.minimum is not None and value < knob.minimum:
        raise ValueError(f"{knob.label} must be >= {knob.minimum}")
    if knob.maximum is not None and value > knob.maximum:
        raise ValueError(f"{knob.label} must be <= {knob.maximum}")
    return value


def _coerce_bool(knob: Knob, raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    raise ValueError(f"{knob.label} must be a boolean (true/false)")


def _split_csv_items(raw: Any) -> tuple[str, ...]:
    if raw is None:
        items: list[str] = []
    elif isinstance(raw, (list, tuple)):
        items = [str(item) for item in raw]
    elif isinstance(raw, str):
        items = raw.split(",")
    else:
        raise ValueError("csv value must be a string, list or tuple")
    cleaned: list[str] = []
    for item in items:
        stripped = item.strip()
        if stripped and stripped not in cleaned:
            cleaned.append(stripped)
    return tuple(cleaned)


def _validate_allowlist_item(knob: Knob, item: str) -> None:
    parts = item.split(":")
    if len(parts) > 3:
        raise ValueError(f"{knob.label}: '{item}' has too many ':'-separated parts")
    family = parts[0]
    if family not in knob.allowed:
        raise ValueError(f"{knob.label}: unknown family '{family}' in '{item}'")
    if len(parts) >= 2:
        side = parts[1]
        if side not in _ALLOWLIST_SIDES:
            raise ValueError(f"{knob.label}: invalid side '{side}' in '{item}'")
    if len(parts) == 3:
        regimes = parts[2].split("|")
        for regime in regimes:
            if regime not in _ALLOWLIST_REGIMES:
                raise ValueError(f"{knob.label}: invalid regime '{regime}' in '{item}'")


def _coerce_csv(knob: Knob, raw: Any) -> tuple[str, ...]:
    items = _split_csv_items(raw)
    if knob.key == "fit_lane_live_allowlist":
        for item in items:
            _validate_allowlist_item(knob, item)
    elif knob.allowed:
        for item in items:
            if item not in knob.allowed:
                raise ValueError(f"{knob.label}: unknown value '{item}' (allowed: {', '.join(knob.allowed)})")
    return items


_COERCERS = {
    "fraction": lambda knob, raw, ceiling: _coerce_fraction(knob, raw),
    "money": lambda knob, raw, ceiling: _coerce_money(knob, raw, ceiling),
    "int": lambda knob, raw, ceiling: _coerce_int(knob, raw),
    "bool": lambda knob, raw, ceiling: _coerce_bool(knob, raw),
    "csv": lambda knob, raw, ceiling: _coerce_csv(knob, raw),
}


def coerce(knob: Knob, raw: Any, *, ceiling: float | None = None) -> Any:
    """Validate and normalise `raw` for `knob`. Raises ValueError on failure."""
    coercer = _COERCERS.get(knob.kind)
    if coercer is None:
        raise ValueError(f"unknown knob kind '{knob.kind}' for {knob.label}")
    return coercer(knob, raw, ceiling)


def _classify_numeric(knob: Knob, old: Any, new: Any) -> str:
    if old is None:
        return "tighten"
    if new == old:
        return "noop"
    increasing = new > old
    if knob.loosen == "increase":
        return "loosen" if increasing else "tighten"
    return "tighten" if increasing else "loosen"


def _classify_bool(knob: Knob, old: Any, new: Any) -> str:
    if old is None:
        return "tighten"
    if new == old:
        return "noop"
    turned_on = new and not old
    if knob.loosen == "true":
        return "loosen" if turned_on else "tighten"
    return "tighten" if turned_on else "loosen"


def _classify_csv(old: Any, new: Any) -> str:
    old_set = set(old) if old else set()
    new_set = set(new) if new else set()
    if old_set == new_set:
        return "noop"
    if new_set - old_set:
        return "loosen"
    return "tighten"


def classify(knob: Knob, old: Any, new: Any) -> str:
    """Classify a change from `old` to `new` as tighten / loosen / noop."""
    if knob.kind in ("fraction", "money", "int"):
        return _classify_numeric(knob, old, new)
    if knob.kind == "bool":
        return _classify_bool(knob, old, new)
    if knob.kind == "csv":
        return _classify_csv(old, new)
    raise ValueError(f"unknown knob kind '{knob.kind}' for {knob.label}")


def env_shadowed(knob: Knob) -> bool:
    """True when an environment variable currently overrides this knob."""
    if not knob.env_name:
        return False
    return os.getenv(knob.env_name) is not None


def _normalise_csv_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return value


def yaml_value(manager: Any, knob: Knob) -> Any:
    """Read the raw yaml-config value for `knob` via `manager.get(...)`."""
    value = manager.get(knob.yaml_path, None)
    if knob.kind == "csv":
        return _normalise_csv_value(value)
    return value


def effective_value(cfg: Any, knob: Knob) -> Any:
    """Read the effective Config property for `knob`, None on any failure."""
    try:
        value = getattr(cfg, knob.key.upper(), None)
    except Exception:
        return None
    if knob.kind == "csv":
        return _normalise_csv_value(value)
    return value
