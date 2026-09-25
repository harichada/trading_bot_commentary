"""v-playbook-desk-b2-2026-09-25 (Task 3): the adaptive playbook — the bot's
memory of which (family, side, regime) setups are actually working.

Three things live here, all of them readable from the live path:

* the value types (`ResolvedTrade`, `PlaybookEntry`, `Playbook`) and their
  JSON store (`load_playbook` never raises; `save_playbook` is atomic),
* `update_entry` — the pure transition rule the nightly updater applies per
  key (`research/strategy_hunt/playbook_update.py` is the only writer), and
* `evaluate_playbook_gate` — the pure decision the router reads, wired in a
  later task and shadow-only until `ENABLE_PLAYBOOK_GATE` is on.

`core` must not import `research`: the updater imports *this* module, never the
other way round. The only status a key can carry is "enabled", "probation" or
"suspended", and the only thing the gate can do to a live signal is skip it
(suspended, flag on) or scale it (`size_mult`, clamped to [0.0, 1.5] on every
path — including read-back, so a hand-edited playbook.json cannot upsize).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core.strategy_fit_gate import GateDecision

logger = logging.getLogger("TradingBot")

__all__ = ["ResolvedTrade", "PlaybookEntry", "Playbook", "PlaybookDecision", "PlaybookKey",
           "clamp_size_mult", "evaluate_playbook_gate", "load_playbook", "playbook_path",
           "recent_stats", "save_playbook", "update_entry"]

PlaybookKey = Tuple[str, str, str]

# The window the *updater* slices per key before calling `update_entry` (newest last).
# NOTE (spec conflict, flagged to the controller): the B2 plan sets this window to 30 and
# the upsize floor `UPSIZE_MIN_N` to 40, so the 1.25 multiplier is unreachable from the
# nightly path. Both numbers are implemented exactly as specified rather than silently
# "fixed"; `update_entry` itself measures whatever list it is handed, so raising this
# window to 40 (or lowering the floor to 30) is a one-constant change.
RECENT_WINDOW = 30
SUSPEND_MIN_N = 10
SUSPEND_MAX_T = -1.0
UPSIZE_MIN_N = 40
UPSIZE_MIN_EXP_R = 0.15
UPSIZE_VERDICTS = ("PASS", "WATCH")

SIZE_MULT_MIN = 0.0
SIZE_MULT_MAX = 1.5
SUSPENDED_SIZE_MULT = 0.0
PROBATION_SIZE_MULT = 0.5
DEFAULT_SIZE_MULT = 1.0
UPSIZE_SIZE_MULT = 1.25

STATUS_ENABLED = "enabled"
STATUS_PROBATION = "probation"
STATUS_SUSPENDED = "suspended"

UNKNOWN_REGIME = "unknown"
FALLBACK_REGIME = "mixed"
KEY_SEP = "|"
DEFAULT_DESK_DIR = Path("data/desk")
PLAYBOOK_FILENAME = "playbook.json"


# ────────────────────────────────────────────────────────────────────
# Coercion helpers — the live loop reads these files, so every read is defensive
# ────────────────────────────────────────────────────────────────────

def _as_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_float(value: Any, default: float = 0.0) -> float:
    """A finite float or `default`: NaN/±inf are "not measured", never written to JSON."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp_size_mult(value: Any) -> float:
    """`value` as a size multiplier inside [SIZE_MULT_MIN, SIZE_MULT_MAX].

    Unreadable or non-finite input clamps to 0.0 (no position), never to 1.0: a
    multiplier nobody can read is not evidence that full size is safe."""
    return min(SIZE_MULT_MAX, max(SIZE_MULT_MIN, _as_float(value, SIZE_MULT_MIN)))


# ────────────────────────────────────────────────────────────────────
# Value types
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResolvedTrade:
    """One closed, R-measured trade for a (family, side, regime) key.

    Produced by the nightly updater from two sources: `source="shadow"` for a
    resolved fit-lane ledger trade, `source="live"` for a `bot_trades` row.
    `exit_ts` is an ISO-8601 string so ordering needs no timezone handling."""
    family: str
    side: str
    regime: str
    r: float
    exit_ts: str
    source: str = "shadow"

    @property
    def key(self) -> PlaybookKey:
        return (self.family, self.side, self.regime)


@dataclass(frozen=True)
class PlaybookEntry:
    family: str
    side: str
    regime: str
    status: str
    size_mult: float
    exp_r_recent: float
    t_stat_recent: float
    n_recent: int
    updated_at: str
    reason: str

    @property
    def key(self) -> PlaybookKey:
        return (self.family, self.side, self.regime)

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "side": self.side, "regime": self.regime, "status": self.status,
                "size_mult": clamp_size_mult(self.size_mult), "exp_r_recent": _as_float(self.exp_r_recent),
                "t_stat_recent": _as_float(self.t_stat_recent), "n_recent": _as_int(self.n_recent),
                "updated_at": self.updated_at, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlaybookEntry":
        """Unknown keys ignored, non-finite floats → 0.0, `size_mult` clamped."""
        status = _as_str(data.get("status"), STATUS_PROBATION)
        return cls(
            family=_as_str(data.get("family")),
            side=_as_str(data.get("side")),
            regime=_as_str(data.get("regime"), UNKNOWN_REGIME),
            status=status if status in (STATUS_ENABLED, STATUS_PROBATION, STATUS_SUSPENDED) else STATUS_PROBATION,
            size_mult=clamp_size_mult(data.get("size_mult")),
            exp_r_recent=_as_float(data.get("exp_r_recent")),
            t_stat_recent=_as_float(data.get("t_stat_recent")),
            n_recent=_as_int(data.get("n_recent")),
            updated_at=_as_str(data.get("updated_at")),
            reason=_as_str(data.get("reason")),
        )


@dataclass(frozen=True)
class Playbook:
    """Immutable map of (family, side, regime) → `PlaybookEntry`.

    Frozen with a plain dict inside: `with_entry` returns a new Playbook rather
    than mutating, so a playbook handed to the router never changes underneath it."""
    entries: Dict[PlaybookKey, PlaybookEntry] = field(default_factory=dict)
    updated_at: str = ""

    def get(self, family: str, side: str, regime: Optional[str]) -> Optional[PlaybookEntry]:
        return self.entries.get((family, side, _as_str(regime, UNKNOWN_REGIME)))

    def with_entry(self, entry: PlaybookEntry) -> "Playbook":
        return Playbook(entries={**self.entries, entry.key: entry}, updated_at=self.updated_at)

    def with_updated_at(self, updated_at: str) -> "Playbook":
        return Playbook(entries=dict(self.entries), updated_at=updated_at)

    def by_status(self, status: str) -> List[PlaybookEntry]:
        return [e for e in self.entries.values() if e.status == status]

    def to_dict(self) -> Dict[str, Any]:
        return {"updated_at": self.updated_at,
                "entries": {KEY_SEP.join(k): e.to_dict() for k, e in sorted(self.entries.items())}}

    @classmethod
    def from_dict(cls, data: Any) -> "Playbook":
        """Never raises: a malformed payload (or one malformed entry) degrades to empty."""
        if not isinstance(data, dict):
            logger.warning("playbook: ignoring a %s payload (expected an object)", type(data).__name__)
            return cls()
        raw = data.get("entries")
        if not isinstance(raw, dict):
            return cls(entries={}, updated_at=_as_str(data.get("updated_at")))
        entries: Dict[PlaybookKey, PlaybookEntry] = {}
        for key, value in raw.items():
            if not isinstance(value, dict):
                logger.warning("playbook: dropping malformed entry %r", key)
                continue
            entry = PlaybookEntry.from_dict(value)
            if not entry.family or not entry.side:
                logger.warning("playbook: dropping entry %r with no family/side", key)
                continue
            entries[entry.key] = entry
        return cls(entries=entries, updated_at=_as_str(data.get("updated_at")))


# ────────────────────────────────────────────────────────────────────
# Store
# ────────────────────────────────────────────────────────────────────

def playbook_path(desk_dir: Any = DEFAULT_DESK_DIR) -> Path:
    return Path(desk_dir) / PLAYBOOK_FILENAME


def load_playbook(path: Any) -> Playbook:
    """The playbook at `path`, or an empty one. Never raises: a missing file is the
    cold-start case and a malformed one must not take the live router down."""
    p = Path(path)
    try:
        if not p.exists():
            return Playbook()
        return Playbook.from_dict(json.loads(p.read_text()))
    except (OSError, ValueError) as exc:
        logger.warning("playbook: could not load %s (%s); using an empty playbook", p, exc)
        return Playbook()


def save_playbook(playbook: Playbook, path: Any) -> Path:
    """Atomic write (tmp + replace) so a reader never sees a half-written playbook."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(playbook.to_dict(), indent=1, allow_nan=False))
        tmp.replace(p)
    except (OSError, ValueError):
        tmp.unlink(missing_ok=True)
        raise
    return p


# ────────────────────────────────────────────────────────────────────
# Rules
# ────────────────────────────────────────────────────────────────────

def recent_stats(recent: Sequence[ResolvedTrade]) -> Tuple[int, float, float]:
    """`(n, exp_r, t_stat)` for `recent`, all finite.

    The t-stat is trade-level (mean / (sample stdev / √n)). It is 0.0 below two
    trades and 0.0 when the sample has no dispersion: identical R values are an
    arithmetic artefact, not a significant result, and must not be able to
    suspend a key on their own."""
    rs = [_as_float(t.r, default=float("nan")) for t in recent]
    rs = [r for r in rs if math.isfinite(r)]
    n = len(rs)
    if n == 0:
        return 0, 0.0, 0.0
    mean = sum(rs) / n
    if n < 2:
        return n, mean, 0.0
    var = sum((r - mean) ** 2 for r in rs) / (n - 1)
    sd = math.sqrt(var)
    if sd <= 0:
        return n, mean, 0.0
    return n, mean, mean / (sd / math.sqrt(n))


def _key_for(prev: Optional[PlaybookEntry], recent: Sequence[ResolvedTrade]) -> PlaybookKey:
    if prev is not None:
        return prev.key
    if recent:
        return recent[-1].key
    raise ValueError("update_entry needs either a previous entry or at least one recent trade to key on")


def update_entry(prev: Optional[PlaybookEntry], recent: Sequence[ResolvedTrade],
                 fit_verdict: Optional[str], now: Any) -> PlaybookEntry:
    """The nightly transition for one key. Pure: `prev` is read, never mutated.

    `recent` is the caller's window (the updater hands over the last
    `RECENT_WINDOW` resolved trades, newest last); every statistic is measured
    on exactly that list. Rule order, first match wins:

    1. suspended  (0.0)  — n ≥ 10, negative expectancy, t < -1: a measured edge
       against us, so no live size at all.
    2. probation  (0.5)  — negative expectancy that isn't yet significant.
    3. probation  (0.5)  — fewer than 10 trades: a cold key trades small.
    4. enabled    (1.25) — exp_R ≥ 0.15 over ≥ 40 trades *and* a PASS/WATCH fit
       verdict: research and the tape agree.
    5. enabled    (1.0)  — everything else.
    """
    family, side, regime = _key_for(prev, recent)
    n, exp_r, t_stat = recent_stats(recent)

    if n >= SUSPEND_MIN_N and exp_r < 0 and t_stat < SUSPEND_MAX_T:
        status, size, reason = STATUS_SUSPENDED, SUSPENDED_SIZE_MULT, "suspended_negative_expectancy"
    elif exp_r < 0:
        status, size, reason = STATUS_PROBATION, PROBATION_SIZE_MULT, "probation_negative_expectancy"
    elif n < SUSPEND_MIN_N:
        status, size, reason = STATUS_PROBATION, PROBATION_SIZE_MULT, "probation_small_sample"
    elif exp_r >= UPSIZE_MIN_EXP_R and n >= UPSIZE_MIN_N and fit_verdict in UPSIZE_VERDICTS:
        status, size, reason = STATUS_ENABLED, UPSIZE_SIZE_MULT, "enabled_upsize_fit_confirmed"
    else:
        status, size, reason = STATUS_ENABLED, DEFAULT_SIZE_MULT, "enabled_normal"

    return PlaybookEntry(family=family, side=side, regime=regime, status=status,
                         size_mult=clamp_size_mult(size), exp_r_recent=round(exp_r, 4),
                         t_stat_recent=round(t_stat, 4), n_recent=n, updated_at=str(now), reason=reason)


# ────────────────────────────────────────────────────────────────────
# Gate
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlaybookDecision(GateDecision):
    """A `GateDecision` (same contract the router already audits) plus the two
    playbook-specific audit fields the risk sizer reads."""
    size_mult: Optional[float] = None
    status: Optional[str] = None

    def audit_fields(self) -> dict:
        out = super().audit_fields()
        if self.size_mult is not None:
            out["size_mult"] = round(float(self.size_mult), 3)
        if self.status is not None:
            out["status"] = self.status
        return out


def _entry_for(playbook: Playbook, family: str, side: str, regime: Optional[str]) -> Optional[PlaybookEntry]:
    """The governing entry: the key's own, or — when there is no regime evidence
    ("unknown"/missing) — the `mixed` entry, which is the closest thing to a
    regime-agnostic measurement the scorer produces."""
    label = _as_str(regime, UNKNOWN_REGIME) or UNKNOWN_REGIME
    entry = playbook.get(family, side, label)
    if entry is None and label == UNKNOWN_REGIME:
        return playbook.get(family, side, FALLBACK_REGIME)
    return entry


def evaluate_playbook_gate(playbook: Playbook, family: str, side: str, regime: Optional[str],
                           enabled: bool) -> GateDecision:
    """What the playbook says about this (family, side, regime).

    `enabled` is `ENABLE_PLAYBOOK_GATE`: with it off a suspended key is only
    audited ("shadow"), never blocked. An unknown family or a key the playbook
    has never measured is explicitly *not* governed — the playbook only ever
    speaks about evidence it has."""
    entry = _entry_for(playbook, family, side, regime) if family and side else None
    if entry is None:
        return PlaybookDecision("allow", "no_playbook_entry", False, family=family or None)

    size = clamp_size_mult(entry.size_mult)
    fields = dict(family=entry.family, regime=entry.regime, status=entry.status, size_mult=size)
    if entry.status == STATUS_SUSPENDED:
        return PlaybookDecision("skip" if enabled else "shadow", "playbook_suspended", True, **fields)
    return PlaybookDecision("allow", f"playbook_{entry.status}", True, **fields)
