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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from core.strategy_fit_gate import GateDecision

logger = logging.getLogger("TradingBot")

__all__ = ["ResolvedTrade", "PlaybookEntry", "Playbook", "PlaybookDecision", "PlaybookKey",
           "carry_forward", "clamp_size_mult", "evaluate_playbook_gate", "load_playbook",
           "playbook_path", "recent_stats", "save_playbook", "update_entry"]

PlaybookKey = Tuple[str, str, str]

# The window the *updater* slices per key before calling `update_entry` (newest last).
# 60 (fix r1, finding 2) so the 40-trade upsize floor is actually reachable; the updater
# also time-bounds the window (`RECENT_DAYS`) before taking the last 60.
RECENT_WINDOW = 60
# A suspended key stops trading, so it can only stay suspended on evidence: after this many
# days with no resolved trade for the key, `carry_forward` puts it back on probation.
SUSPENSION_LAPSE_DAYS = 30
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
# Fallback only, for the research CLI and tests. The live path (Task 5) must pass
# `Config().DESK_DATA_DIR` explicitly — this module never reads Config, so a deployment that
# moves the desk dir would otherwise silently read a playbook nobody writes.
DEFAULT_DESK_DIR = Path("data/desk")
PLAYBOOK_FILENAME = "playbook.json"


# ────────────────────────────────────────────────────────────────────
# Coercion helpers — the live loop reads these files, so every read is defensive
# ────────────────────────────────────────────────────────────────────

def _as_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_float(value: Any, default: float = 0.0) -> float:
    """A finite float or `default`: NaN/±inf coerce, so nothing unreadable reaches the rules."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _json_float(value: Any) -> Optional[float]:
    """A finite float, or None for anything unreadable or non-finite: a t-stat is ±inf when a
    sample has no dispersion (fix r1, finding 4) and JSON has no Infinity literal, so `null` is
    the honest wire value (the nightly recompute restores the real number)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


# ── Value types ──

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
                "size_mult": clamp_size_mult(self.size_mult), "exp_r_recent": _json_float(self.exp_r_recent),
                "t_stat_recent": _json_float(self.t_stat_recent), "n_recent": _as_int(self.n_recent),
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


# ── Store ──

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


# ── Rules ──

def recent_stats(recent: Sequence[ResolvedTrade]) -> Tuple[int, float, float]:
    """`(n, exp_r, t_stat)` for `recent`; `exp_r` is always finite, `t_stat` may be ±inf.

    The t-stat is trade-level (mean / (sample stdev / √n)). It is 0.0 below two trades, and
    with no dispersion (fix r1, finding 4) it is ±inf for a non-zero mean — twelve identical
    -1R stop-outs are the *most* certain losing sample there is and must suspend — and 0.0
    for a zero mean, where there is nothing to be certain about."""
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
        return n, mean, 0.0 if mean == 0 else math.copysign(float("inf"), mean)
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
                         # round(±inf, 4) is ±inf — kept as-is in memory, `null` on disk.
                         t_stat_recent=round(t_stat, 4), n_recent=n, updated_at=str(now), reason=reason)


def carry_forward(prev: PlaybookEntry, now: Any, has_recent_trades: bool) -> PlaybookEntry:
    """`prev` as tonight's entry for a key with no new measurement — unchanged, except that a
    suspension lapses to probation once the key has gone `SUSPENSION_LAPSE_DAYS` without a
    resolved trade (fix r1, finding 3): a suspended key places no live orders, so "no trades"
    is the expected state, not continuing evidence against it. The numbers that suspended it
    are carried over so the journal can still explain the lapse."""
    if prev.status != STATUS_SUSPENDED or has_recent_trades:
        return prev
    return PlaybookEntry(family=prev.family, side=prev.side, regime=prev.regime,
                         status=STATUS_PROBATION, size_mult=PROBATION_SIZE_MULT,
                         exp_r_recent=prev.exp_r_recent, t_stat_recent=prev.t_stat_recent,
                         n_recent=prev.n_recent, updated_at=str(now),
                         reason="suspension_lapsed_no_recent_trades")


# ── Gate ──

@dataclass(frozen=True)
class PlaybookDecision(GateDecision):
    """A `GateDecision` (same contract the router already audits) plus the two
    playbook-specific audit fields the risk sizer reads."""
    size_mult: Optional[float] = None
    status: Optional[str] = None
    playbook_fallback: Optional[str] = None

    def audit_fields(self) -> dict:
        out = super().audit_fields()
        if self.size_mult is not None:
            out["size_mult"] = round(float(self.size_mult), 3)
        if self.status is not None:
            out["status"] = self.status
        if self.playbook_fallback is not None:
            out["playbook_fallback"] = self.playbook_fallback
        return out


def _entry_for(playbook: Playbook, family: str, side: str,
               regime: Optional[str]) -> Tuple[Optional[PlaybookEntry], Optional[str]]:
    """`(entry, fallback_label)` — the governing entry and which fallback found it.

    Lookup order (fix r1, finding 1b): the exact regime; then, only when the live regime is
    unknown, the `mixed` entry (a measured regime may never be governed by another one); then
    the `unknown` entry as an all-regimes fallback, which is the only measurement most keys
    have while the shadow tape carries no regime."""
    label = _as_str(regime, UNKNOWN_REGIME) or UNKNOWN_REGIME
    exact = playbook.get(family, side, label)
    if exact is not None:
        return exact, None
    if label == UNKNOWN_REGIME:
        mixed = playbook.get(family, side, FALLBACK_REGIME)
        return (mixed, FALLBACK_REGIME) if mixed is not None else (None, None)
    unknown = playbook.get(family, side, UNKNOWN_REGIME)
    return (unknown, UNKNOWN_REGIME) if unknown is not None else (None, None)


def evaluate_playbook_gate(playbook: Playbook, family: str, side: str, regime: Optional[str],
                           enabled: bool) -> GateDecision:
    """What the playbook says about this (family, side, regime).

    `enabled` is `ENABLE_PLAYBOOK_GATE`: with it off a suspended key is only
    audited ("shadow"), never blocked. An unknown family or a key the playbook
    has never measured is explicitly *not* governed — the playbook only ever
    speaks about evidence it has."""
    entry, fallback = _entry_for(playbook, family, side, regime) if family and side else (None, None)
    if entry is None:
        return PlaybookDecision("allow", "no_playbook_entry", False, family=family or None)

    size = clamp_size_mult(entry.size_mult)
    fields = dict(family=entry.family, regime=entry.regime, status=entry.status, size_mult=size,
                  playbook_fallback=fallback)
    if entry.status == STATUS_SUSPENDED:
        return PlaybookDecision("skip" if enabled else "shadow", "playbook_suspended", True, **fields)
    return PlaybookDecision("allow", f"playbook_{entry.status}", True, **fields)
