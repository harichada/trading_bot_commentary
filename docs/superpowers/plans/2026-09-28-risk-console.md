# Risk Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the go-live risk knobs and lane switches editable from the dashboard Settings tab with industry-standard semantics: an instant "block new entries" switch, tighten-now / loosen-at-next-open ratchet, validation, and an audit trail.

**Architecture:** A pure module `core/settings_console.py` owns a knob registry (yaml path, type, bounds, which direction is "loosening", restart-required), classifies each change, applies tightening immediately through the existing `ConfigManager.update`, queues loosening in `<DESK_DATA_DIR>/pending_settings.json` for the desk loop to promote at 09:25 ET, and appends every event to `<DESK_DATA_DIR>/settings_audit.ndjson`. A FastAPI router `api/console.py` exposes it; the dashboard Settings tab gets a "Risk Console" card; the engine's signal router honours `Config().BLOCK_NEW_ENTRIES`. A migration script moves today's `.env` go-live values into `Config().yaml` so the API is the source of truth.

**Tech Stack:** Python 3.11, FastAPI, PyYAML, pytest; vanilla JS in `templates/dashboard.html`.

**Spec:** the design agreed in chat on 2026-09-28 (this section restates it):
1. "Block new entries" on the dashboard, instant, audited. Unblock is instant but needs `confirm: "YES"`.
2. Risk knobs editable live with the ratchet: a tightening applies now; a loosening is queued and applied at the next session open (09:25 ET).
3. Lane switches: disable is instant, enable waits for next open.
4. `DESK_STREAK_SINCE` and the allocation ceiling are file-only (not in the console). `FIT_LANES_ENABLED` is shown read-only with a "restart required" badge.
5. One audit row per change; validation refuses fractions outside (0, 1], allocation above the ceiling, unknown lanes.
6. `.env` values shadow yaml values (env-first precedence in `core/config.py`). The console must say so (`env_shadowed`) and refuse writes to shadowed knobs with HTTP 409 until the migration removes them from `.env`.

## Global Constraints

- Branch `feature/newsbus-autonomy-decision-card` only. Never push. Never commit `.env`, `Config().yaml`, `trading_state.json`, `trading_bot.log`, `trading_brain.json`, `trading_commentary.json`, `data/`, `models/`. Commit with explicit paths.
- Commit trailer on every commit:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01BT1pvoczNh7y1pBxVN53R9`.
- No secrets in code, tests, docs or logs. Never print `.env` contents.
- The live bot (pid 1301804, port 9000) is trading. Do NOT restart it, call `/api/stop`, `/api/toggle-mode`, or write `Config().yaml` / `.env` of the repo root from any test. Tests use `tmp_path` for every file they touch, and a fresh `ConfigManager(config_path=<tmp>)` instance, never the module-level `config_manager`.
- All new files ≤ 400 lines; functions < 50 lines. Immutable style: return new dicts/dataclasses rather than mutating inputs (the one sanctioned mutation is `ConfigManager.update`, which is the persistence boundary).
- Every step is `never raises` at the API/loop boundary: log + degrade, mirroring `/api/desk/*` (`{ok: false, error}` with HTTP 200 for reads; explicit 400/409 for rejected writes).
- Tests: pytest, run with `/home/nvidia/anaconda3/envs/trading-bot/bin/python -m pytest <file> -q`. Existing suite must stay green for the files touched (`tests/core/desk`, `tests/api`, `tests/test_config.py`).
- Time is ET (`ZoneInfo("America/New_York")`); the desk helpers in `core/loops/desk_steps.py` (`now_et`, `is_session_day`, `session_date_str`, `crossed`, `parse_et_time`) are the only clock helpers to use.

---

### Task 1: Knob registry, validation and change classification

**Files:**
- Create: `core/settings_console.py`
- Test: `tests/core/test_settings_console_registry.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Knob:
      key: str            # console key, e.g. "risk_per_trade_pct"
      yaml_path: str      # e.g. "trading.risk_per_trade_pct"
      env_name: str       # e.g. "RISK_PER_TRADE_PCT" ("" when no env override exists)
      kind: str           # "fraction" | "money" | "int" | "bool" | "csv"
      group: str          # "risk" | "lanes" | "gates"
      label: str
      loosen: str         # "increase" | "decrease" | "true" | "false" | "add" — which direction is loosening
      restart_required: bool = False
      minimum: float | None = None
      maximum: float | None = None
      allowed: tuple[str, ...] = ()   # csv kinds: allowed items (families / "family:side:regime" prefixes)

  KNOBS: tuple[Knob, ...]
  KNOB_BY_KEY: dict[str, Knob]
  ALLOCATION_CEILING_PATH = "trading.settings_console.allocation_ceiling"
  DEFAULT_ALLOCATION_CEILING = 5000.0
  KNOWN_FIT_FAMILIES = ("mr_20d", "mom_12_1", "dt_momentum", "gap_play", "vwap_reversion", "mean_rev_intraday", "low_vol_trend", "orb_inplay", "pead_smallcap", "rs_leader")

  def coerce(knob: Knob, raw: Any, *, ceiling: float | None = None) -> Any            # raises ValueError with a user-readable message
  def classify(knob: Knob, old: Any, new: Any) -> str                                  # "tighten" | "loosen" | "noop"
  def env_shadowed(knob: Knob) -> bool                                                 # os.getenv(knob.env_name) is not None
  def yaml_value(manager: Any, knob: Knob) -> Any                                      # manager.get(knob.yaml_path, None)
  def effective_value(cfg: Any, knob: Knob) -> Any                                     # Config property by knob.key.upper(), None on failure
  ```

Registry contents (exact):

| key | yaml_path | env_name | kind | group | loosen | min/max | notes |
|---|---|---|---|---|---|---|---|
| bot_capital_allocation | trading.bot_capital_allocation | BOT_CAPITAL_ALLOCATION | money | risk | increase | >0, ≤ ceiling | ceiling from `ALLOCATION_CEILING_PATH` |
| risk_per_trade_pct | trading.risk_per_trade_pct | RISK_PER_TRADE_PCT | fraction | risk | increase | (0,1] | |
| max_daily_loss | trading.max_daily_loss | MAX_DAILY_LOSS | fraction | risk | increase | (0,1] | |
| max_positions | trading.max_positions | "" | int | risk | increase | 1..50 | |
| swing_max_hold_sessions | trading.swing_max_hold_sessions | SWING_MAX_HOLD_SESSIONS | int | risk | increase | 1..60 | |
| orb_live_entries_enabled | trading.orb_live_entries_enabled | ORB_LIVE_ENTRIES_ENABLED | bool | lanes | true | | |
| day_trade_live_entries_enabled | trading.day_trade_live_entries_enabled | DAY_TRADE_LIVE_ENTRIES_ENABLED | bool | lanes | true | | |
| day_trade_short_live_entries_enabled | trading.day_trade_short_live_entries_enabled | DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED | bool | lanes | true | | |
| mean_rev_live_entries_enabled | trading.mean_rev_live_entries_enabled | MEAN_REV_LIVE_ENTRIES_ENABLED | bool | lanes | true | | |
| fit_lane_live_entries_enabled | trading.fit_lane_live_entries_enabled | FIT_LANE_LIVE_ENTRIES_ENABLED | csv | lanes | add | | allowed = KNOWN_FIT_FAMILIES |
| fit_lane_live_allowlist | trading.fit_lane_live_allowlist | FIT_LANE_LIVE_ALLOWLIST | csv | lanes | add | | each item `family[:side[:regimes]]`, family in KNOWN_FIT_FAMILIES, side in {long, short}, regimes `|`-joined subset of {risk_on, risk_off, mixed} |
| fit_lanes_enabled | trading.fit_lanes_enabled | FIT_LANES_ENABLED | csv | lanes | add | | restart_required=True (read-only in UI; API returns 409 `restart_required`) |
| enable_playbook_gate | trading.enable_playbook_gate | ENABLE_PLAYBOOK_GATE | bool | gates | false | | turning a gate off is loosening |
| enable_desk_limits | trading.enable_desk_limits | ENABLE_DESK_LIMITS | bool | gates | false | | |

Classification rules:
- `noop` when coerced old == coerced new (csv compared as sets).
- numeric `loosen="increase"`: new > old → `loosen`, new < old → `tighten`. bool `loosen="true"`: old False→True is `loosen`, True→False `tighten`; `loosen="false"` inverse. csv `loosen="add"`: any item in new not in old → `loosen`; otherwise (pure removal) `tighten`.
- `old is None` (no yaml value yet): treat as `loosen` for money/fraction/int only when new exceeds the Config default the property would return (pass `old=effective_value(cfg, knob)` from the caller; the registry function itself just compares old/new and treats `None` old as `tighten` — the caller is responsible for supplying the effective old value).

- [ ] **Step 1: Write the failing tests** (`tests/core/test_settings_console_registry.py`)

```python
import os
import pytest
from core import settings_console as sc

def test_registry_keys_unique_and_complete():
    keys = [k.key for k in sc.KNOBS]
    assert len(keys) == len(set(keys))
    for expected in ("bot_capital_allocation", "risk_per_trade_pct", "max_daily_loss", "max_positions",
                     "swing_max_hold_sessions", "orb_live_entries_enabled", "day_trade_live_entries_enabled",
                     "day_trade_short_live_entries_enabled", "mean_rev_live_entries_enabled",
                     "fit_lane_live_entries_enabled", "fit_lane_live_allowlist", "fit_lanes_enabled",
                     "enable_playbook_gate", "enable_desk_limits"):
        assert expected in sc.KNOB_BY_KEY
    assert sc.KNOB_BY_KEY["fit_lanes_enabled"].restart_required is True
    assert "desk_streak_since" not in sc.KNOB_BY_KEY

@pytest.mark.parametrize("raw", ["0", "1.5", "-0.1", "abc", ""])
def test_fraction_rejects_out_of_range(raw):
    with pytest.raises(ValueError):
        sc.coerce(sc.KNOB_BY_KEY["risk_per_trade_pct"], raw)

def test_fraction_accepts_edge():
    assert sc.coerce(sc.KNOB_BY_KEY["risk_per_trade_pct"], "1") == 1.0
    assert sc.coerce(sc.KNOB_BY_KEY["max_daily_loss"], 0.03) == 0.03

def test_money_respects_ceiling():
    knob = sc.KNOB_BY_KEY["bot_capital_allocation"]
    assert sc.coerce(knob, "4500", ceiling=5000) == 4500.0
    with pytest.raises(ValueError):
        sc.coerce(knob, 5001, ceiling=5000)
    with pytest.raises(ValueError):
        sc.coerce(knob, 0, ceiling=5000)

def test_int_bounds():
    assert sc.coerce(sc.KNOB_BY_KEY["max_positions"], "5") == 5
    with pytest.raises(ValueError):
        sc.coerce(sc.KNOB_BY_KEY["max_positions"], 0)
    with pytest.raises(ValueError):
        sc.coerce(sc.KNOB_BY_KEY["max_positions"], "2.5")

@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("on", True), (False, False), ("0", False), ("no", False)])
def test_bool_coercion(raw, expected):
    assert sc.coerce(sc.KNOB_BY_KEY["orb_live_entries_enabled"], raw) is expected

def test_bool_rejects_garbage():
    with pytest.raises(ValueError):
        sc.coerce(sc.KNOB_BY_KEY["orb_live_entries_enabled"], "maybe")

def test_csv_families_known_only():
    knob = sc.KNOB_BY_KEY["fit_lane_live_entries_enabled"]
    assert sc.coerce(knob, "mr_20d, mom_12_1") == ("mr_20d", "mom_12_1")
    assert sc.coerce(knob, ["mr_20d"]) == ("mr_20d",)
    assert sc.coerce(knob, "") == ()
    with pytest.raises(ValueError):
        sc.coerce(knob, "mr_20d,bogus")

def test_allowlist_grammar():
    knob = sc.KNOB_BY_KEY["fit_lane_live_allowlist"]
    assert sc.coerce(knob, "mr_20d:long,mom_12_1:long:risk_on|mixed") == ("mr_20d:long", "mom_12_1:long:risk_on|mixed")
    for bad in ("mr_20d:sideways", "mom_12_1:long:bull", "bogus:long", "mr_20d:long:risk_on:extra"):
        with pytest.raises(ValueError):
            sc.coerce(knob, bad)

def test_classify_numeric():
    knob = sc.KNOB_BY_KEY["risk_per_trade_pct"]
    assert sc.classify(knob, 0.01, 0.005) == "tighten"
    assert sc.classify(knob, 0.01, 0.02) == "loosen"
    assert sc.classify(knob, 0.01, 0.01) == "noop"
    assert sc.classify(knob, None, 0.01) == "tighten"

def test_classify_bool_lane_and_gate():
    lane = sc.KNOB_BY_KEY["orb_live_entries_enabled"]
    assert sc.classify(lane, False, True) == "loosen"
    assert sc.classify(lane, True, False) == "tighten"
    gate = sc.KNOB_BY_KEY["enable_playbook_gate"]
    assert sc.classify(gate, True, False) == "loosen"
    assert sc.classify(gate, False, True) == "tighten"

def test_classify_csv():
    knob = sc.KNOB_BY_KEY["fit_lane_live_entries_enabled"]
    assert sc.classify(knob, ("mr_20d",), ("mr_20d", "mom_12_1")) == "loosen"
    assert sc.classify(knob, ("mr_20d", "mom_12_1"), ("mr_20d",)) == "tighten"
    assert sc.classify(knob, ("mr_20d", "mom_12_1"), ("mom_12_1", "mr_20d")) == "noop"
    assert sc.classify(knob, ("mr_20d",), ("mom_12_1",)) == "loosen"   # swap adds one

def test_env_shadowed(monkeypatch):
    knob = sc.KNOB_BY_KEY["risk_per_trade_pct"]
    monkeypatch.delenv("RISK_PER_TRADE_PCT", raising=False)
    assert sc.env_shadowed(knob) is False
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "0.01")
    assert sc.env_shadowed(knob) is True
    assert sc.env_shadowed(sc.KNOB_BY_KEY["max_positions"]) is False   # no env name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/nvidia/anaconda3/envs/trading-bot/bin/python -m pytest tests/core/test_settings_console_registry.py -q`
Expected: FAIL with `ModuleNotFoundError: core.settings_console`.

- [ ] **Step 3: Implement `core/settings_console.py`** (registry + coerce + classify + env_shadowed + yaml_value + effective_value). Keep the module header docstring short and state the ratchet rule. `coerce` dispatches on `knob.kind` to small private functions (`_coerce_fraction`, `_coerce_money`, `_coerce_int`, `_coerce_bool`, `_coerce_csv`). `_coerce_csv` accepts a comma string, list or tuple, strips, drops empties, de-duplicates preserving order, validates each item against `knob.allowed` (families) or the allowlist grammar when `knob.key == "fit_lane_live_allowlist"`. Bool accepts `bool`, and strings `1/true/yes/on` vs `0/false/no/off` case-insensitively; anything else raises. Error messages name the knob label and the accepted range.

- [ ] **Step 4: Run tests to verify they pass**

Run: same command. Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add core/settings_console.py tests/core/test_settings_console_registry.py
git commit -m "feat(console): knob registry, validation and tighten/loosen classification"
```

---

### Task 2: Change submission, pending queue, promotion and audit

**Files:**
- Create: `core/settings_console_store.py`
- Test: `tests/core/test_settings_console_store.py`

**Interfaces:**
- Consumes: Task 1 (`Knob`, `KNOB_BY_KEY`, `coerce`, `classify`, `env_shadowed`, `yaml_value`, `ALLOCATION_CEILING_PATH`, `DEFAULT_ALLOCATION_CEILING`).
- Produces:
  ```python
  PENDING_FILE = "pending_settings.json"
  AUDIT_FILE = "settings_audit.ndjson"
  PROMOTE_TIME_ET = (9, 25)

  @dataclass(frozen=True)
  class ChangeResult:
      status: str            # "applied" | "pending" | "rejected" | "noop"
      key: str
      old: Any
      new: Any
      direction: str         # "tighten" | "loosen" | "noop" | ""
      effective_at: str      # ISO ET timestamp; "" when applied/rejected
      pending_id: str        # "" unless status == "pending"
      reason: str            # user-supplied reason, or the rejection message

  def next_promotion(now: datetime) -> datetime         # today 09:25 if now < 09:25 on a session day, else next session day 09:25 (weekday-only, matches desk_steps.is_session_day)
  def submit_change(manager, cfg, desk_dir: str, key: str, raw_value: Any, *, now: datetime, actor: str, reason: str) -> ChangeResult
  def list_pending(desk_dir: str) -> list[dict]
  def cancel_pending(desk_dir: str, pending_id: str, *, now: datetime, actor: str) -> bool
  def promote_pending(manager, desk_dir: str, *, now: datetime) -> list[dict]     # applies every pending row with effective_at <= now; returns the applied rows
  def read_audit(desk_dir: str, limit: int = 50) -> list[dict]
  def set_block_new_entries(manager, desk_dir: str, block: bool, *, now: datetime, actor: str, reason: str, confirm: str = "") -> ChangeResult
  ```
- Pending row shape (JSON): `{"id": "<uuid4 hex[:12]>", "key", "old", "new", "submitted_at", "effective_at", "actor", "reason"}`. Pending file is a JSON list; writes are atomic (write `<file>.tmp` then `os.replace`).
- Audit row shape (NDJSON, one per line): `{"ts", "actor", "event": "applied"|"pending"|"promoted"|"cancelled"|"rejected"|"block_entries"|"unblock_entries", "key", "old", "new", "direction", "effective_at", "reason"}`.
- `submit_change` rules, in order:
  1. unknown key → `rejected`, reason `unknown setting`.
  2. `knob.restart_required` → `rejected`, reason `restart_required`.
  3. `env_shadowed(knob)` → `rejected`, reason `env_shadowed: <ENV_NAME> is set in .env; the yaml value is ignored until it is removed`.
  4. `coerce` (ceiling = `manager.get(ALLOCATION_CEILING_PATH, DEFAULT_ALLOCATION_CEILING)`) → on ValueError `rejected` with its message.
  5. `old = effective_value(cfg, knob)`, fall back to `yaml_value` when None. `direction = classify(...)`. `noop` → status `noop`, no write, no audit.
  6. `tighten` → `manager.update(knob.yaml_path, stored)` where `stored` is `list(new)` for csv kinds and the coerced scalar otherwise; audit `applied`.
  7. `loosen` → append pending row with `effective_at = next_promotion(now)`; if a pending row for the same key exists it is replaced (one pending per key); audit `pending`.
- `promote_pending` applies rows with `effective_at <= now` via `manager.update`, audits `promoted`, rewrites the pending file with the remainder. Never raises: a row that fails to apply is kept and logged.
- `set_block_new_entries`: `block=True` → `manager.update("trading.block_new_entries", True)`, audit `block_entries`, status `applied`. `block=False` requires `confirm == "YES"` else `rejected` (`confirm required`); on success writes False and audits `unblock_entries`. Both always immediate.
- All file ops use `desk_dir` (`Config().DESK_DATA_DIR` from the caller); create the directory if missing.

- [ ] **Step 1: Write the failing tests** (`tests/core/test_settings_console_store.py`) — use a `FakeManager` with `get`/`update` over a dict (dot paths) and a `FakeCfg` exposing upper-case properties for the effective values:

```python
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import pytest
from core import settings_console_store as store

ET = ZoneInfo("America/New_York")

class FakeManager:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.updates = []
    def get(self, key, default=None):
        return self.data.get(key, default)
    def update(self, key, value):
        self.data[key] = value
        self.updates.append((key, value))

class FakeCfg:
    RISK_PER_TRADE_PCT = 0.01
    BOT_CAPITAL_ALLOCATION = 5000.0
    ORB_LIVE_ENTRIES_ENABLED = True
    FIT_LANE_LIVE_ENTRIES_ENABLED = ("mr_20d", "mom_12_1")
    MAX_POSITIONS = 5
    ENABLE_PLAYBOOK_GATE = True

def _mon(h=10, m=0):  # Monday 2026-09-28
    return datetime(2026, 9, 28, h, m, tzinfo=ET)

@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for name in ("RISK_PER_TRADE_PCT", "BOT_CAPITAL_ALLOCATION", "ORB_LIVE_ENTRIES_ENABLED",
                 "FIT_LANE_LIVE_ENTRIES_ENABLED", "ENABLE_PLAYBOOK_GATE", "FIT_LANES_ENABLED"):
        monkeypatch.delenv(name, raising=False)

def test_next_promotion_same_day_before_925():
    assert store.next_promotion(_mon(8, 0)) == _mon(9, 25)

def test_next_promotion_after_925_rolls_to_next_session_day():
    assert store.next_promotion(_mon(10, 0)) == datetime(2026, 9, 29, 9, 25, tzinfo=ET)
    fri = datetime(2026, 10, 2, 15, 0, tzinfo=ET)
    assert store.next_promotion(fri) == datetime(2026, 10, 5, 9, 25, tzinfo=ET)

def test_tighten_applies_immediately_and_audits(tmp_path):
    m = FakeManager()
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", "0.005",
                            now=_mon(), actor="test", reason="cut risk")
    assert r.status == "applied" and r.direction == "tighten"
    assert m.data["trading.risk_per_trade_pct"] == 0.005
    rows = store.read_audit(str(tmp_path))
    assert rows[-1]["event"] == "applied" and rows[-1]["key"] == "risk_per_trade_pct"

def test_loosen_queues_until_next_open(tmp_path):
    m = FakeManager()
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.02,
                            now=_mon(), actor="test", reason="more risk")
    assert r.status == "pending" and r.direction == "loosen"
    assert "trading.risk_per_trade_pct" not in m.data
    pend = store.list_pending(str(tmp_path))
    assert len(pend) == 1 and pend[0]["key"] == "risk_per_trade_pct" and pend[0]["id"] == r.pending_id
    assert pend[0]["effective_at"].startswith("2026-09-29T09:25")

def test_second_loosen_replaces_pending_for_same_key(tmp_path):
    m = FakeManager()
    store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.02, now=_mon(), actor="t", reason="")
    store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.03, now=_mon(), actor="t", reason="")
    pend = store.list_pending(str(tmp_path))
    assert len(pend) == 1 and pend[0]["new"] == 0.03

def test_promote_applies_due_rows_only(tmp_path):
    m = FakeManager()
    store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.02, now=_mon(), actor="t", reason="")
    assert store.promote_pending(m, str(tmp_path), now=datetime(2026, 9, 29, 9, 0, tzinfo=ET)) == []
    applied = store.promote_pending(m, str(tmp_path), now=datetime(2026, 9, 29, 9, 26, tzinfo=ET))
    assert [a["key"] for a in applied] == ["risk_per_trade_pct"]
    assert m.data["trading.risk_per_trade_pct"] == 0.02
    assert store.list_pending(str(tmp_path)) == []
    assert store.read_audit(str(tmp_path))[-1]["event"] == "promoted"

def test_cancel_pending(tmp_path):
    m = FakeManager()
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "orb_live_entries_enabled", True, now=_mon(), actor="t", reason="")
    # ORB already True in FakeCfg → noop; use a real loosen instead
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "max_positions", 8, now=_mon(), actor="t", reason="")
    assert r.status == "pending"
    assert store.cancel_pending(str(tmp_path), r.pending_id, now=_mon(), actor="t") is True
    assert store.list_pending(str(tmp_path)) == []
    assert store.cancel_pending(str(tmp_path), "nope", now=_mon(), actor="t") is False

def test_csv_stored_as_list_and_removal_is_tighten(tmp_path):
    m = FakeManager()
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "fit_lane_live_entries_enabled", "mr_20d",
                            now=_mon(), actor="t", reason="drop momentum")
    assert r.status == "applied" and m.data["trading.fit_lane_live_entries_enabled"] == ["mr_20d"]

def test_rejections(tmp_path, monkeypatch):
    m = FakeManager()
    assert store.submit_change(m, FakeCfg(), str(tmp_path), "nope", 1, now=_mon(), actor="t", reason="").status == "rejected"
    assert store.submit_change(m, FakeCfg(), str(tmp_path), "fit_lanes_enabled", "mr_20d", now=_mon(), actor="t", reason="").reason == "restart_required"
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "0.01")
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.005, now=_mon(), actor="t", reason="")
    assert r.status == "rejected" and r.reason.startswith("env_shadowed")
    monkeypatch.delenv("RISK_PER_TRADE_PCT")
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "bot_capital_allocation", 9000, now=_mon(), actor="t", reason="")
    assert r.status == "rejected"
    assert store.read_audit(str(tmp_path))[-1]["event"] == "rejected"

def test_block_and_unblock(tmp_path):
    m = FakeManager()
    r = store.set_block_new_entries(m, str(tmp_path), True, now=_mon(), actor="t", reason="glitch")
    assert r.status == "applied" and m.data["trading.block_new_entries"] is True
    r = store.set_block_new_entries(m, str(tmp_path), False, now=_mon(), actor="t", reason="")
    assert r.status == "rejected" and m.data["trading.block_new_entries"] is True
    r = store.set_block_new_entries(m, str(tmp_path), False, now=_mon(), actor="t", reason="ok", confirm="YES")
    assert r.status == "applied" and m.data["trading.block_new_entries"] is False
    events = [row["event"] for row in store.read_audit(str(tmp_path))]
    assert events[-2:] == ["rejected", "unblock_entries"] or events[-1] == "unblock_entries"

def test_pending_file_corruption_degrades_to_empty(tmp_path):
    (tmp_path / store.PENDING_FILE).write_text("{not json")
    assert store.list_pending(str(tmp_path)) == []
```

- [ ] **Step 2: Run to verify failure.** Expected: `ModuleNotFoundError`.
- [ ] **Step 3: Implement `core/settings_console_store.py`** per the interface. Use `uuid.uuid4().hex[:12]` for ids, `datetime.isoformat()` in ET for timestamps, `json` for files, atomic writes, `logging.getLogger(__name__)`. `effective_value` from Task 1 supplies `old`; when it returns None use `yaml_value`.
- [ ] **Step 4: Run tests; all PASS.** Also run `tests/core/test_settings_console_registry.py`.
- [ ] **Step 5: Commit**

```bash
git add core/settings_console_store.py tests/core/test_settings_console_store.py
git commit -m "feat(console): ratchet submission, pending queue, promotion and audit trail"
```

---

### Task 3: Engine block switch, config properties, desk-loop promotion step

**Files:**
- Modify: `core/config.py` (add two properties next to `ENABLE_DESK_LIMITS`, ~line 4376)
- Modify: `core/engine.py:6423` (`_process_signal_with_commentary`, right after the `received` audit)
- Modify: `core/loops/desk_loop.py` (`_schedule`, new `_step_settings`)
- Test: `tests/test_config.py` (append), `tests/core/desk/test_desk_loop_settings_step.py` (new), `tests/core/test_engine_block_entries.py` (new)

**Interfaces:**
- Produces on `Config`:
  ```python
  @property
  def BLOCK_NEW_ENTRIES(self) -> bool:      # yaml-only: trading.block_new_entries, default False. No env override — the console owns it.
  @property
  def SETTINGS_PROMOTE_TIME_ET(self) -> str: # env SETTINGS_PROMOTE_TIME_ET or yaml trading.settings_promote_time_et, default "09:25"
  ```
- Engine: immediately after the existing `received` audit in `_process_signal_with_commentary`:
  ```python
  # v-risk-console-2026-09-28: operator kill switch for NEW entries only.
  # Exits and position management never pass through this router.
  if bool(getattr(Config(), "BLOCK_NEW_ENTRIES", False)):
      self._audit("signal_router", signal.symbol, "skip", "entries_blocked_by_console")
      return
  ```
  (Confirm `Config` is already imported in `core/engine.py`; it is, via `from core.config import Config`. If the name differs, use the module's existing alias.)
- Desk loop: add `("settings", steps.parse_et_time(cfg.SETTINGS_PROMOTE_TIME_ET, "09:25"), self._step_settings)` as the FIRST schedule entry, and:
  ```python
  async def _step_settings(self, cfg: Any, now: datetime, date: str) -> None:
      from core.config import config_manager
      from core.settings_console_store import promote_pending
      applied = await asyncio.to_thread(promote_pending, config_manager, cfg.DESK_DATA_DIR, now=now)
      for row in applied:
          self._audit("settings_console", "promoted", str(row.get("key", "")),
                      old=row.get("old"), new=row.get("new"), effective_at=row.get("effective_at"))
  ```
  Tests inject `config_fn` returning a stub with `DESK_DATA_DIR` and `SETTINGS_PROMOTE_TIME_ET`, and monkeypatch `core.loops.desk_loop.promote_pending`? — No: to keep the import local and patchable, import at module top instead: `from core.settings_console_store import promote_pending` and `from core.config import Config, config_manager`; tests monkeypatch `desk_loop.promote_pending` and `desk_loop.config_manager`.

- [ ] **Step 1: Tests.**

`tests/test_config.py` (append):
```python
def test_block_new_entries_yaml_only(tmp_path, monkeypatch):
    from core.config import Config, ConfigManager
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"trading": {"block_new_entries": True}}))
    cfg = Config()
    monkeypatch.setattr(cfg, "manager", ConfigManager(config_path=str(p)))
    monkeypatch.setenv("BLOCK_NEW_ENTRIES", "0")   # env must NOT override
    assert cfg.BLOCK_NEW_ENTRIES is True
    cfg.manager.update("trading.block_new_entries", False)
    assert cfg.BLOCK_NEW_ENTRIES is False

def test_settings_promote_time_default_and_env(monkeypatch):
    from core.config import Config
    monkeypatch.delenv("SETTINGS_PROMOTE_TIME_ET", raising=False)
    assert Config().SETTINGS_PROMOTE_TIME_ET == "09:25"
    monkeypatch.setenv("SETTINGS_PROMOTE_TIME_ET", "09:31")
    assert Config().SETTINGS_PROMOTE_TIME_ET == "09:31"
```
(Check how existing tests in `tests/test_config.py` build a `Config` against a temp yaml and follow that pattern if one exists; `ConfigManager._validate_config` may require keys — if so write the minimal yaml it needs.)

`tests/core/desk/test_desk_loop_settings_step.py`:
```python
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
from core.loops import desk_loop as dl

ET = ZoneInfo("America/New_York")

class _Engine:
    def __init__(self):
        self.audits = []
    def _audit(self, component, symbol, action, reason, /, **details):
        self.audits.append((component, action, reason, details))

def _cfg(tmp_path):
    return SimpleNamespace(DESK_DATA_DIR=str(tmp_path), SETTINGS_PROMOTE_TIME_ET="09:25",
                           DESK_PLAN_TIME_ET="09:00", DESK_REFRESH_TIME_ET="09:35", DESK_JOURNAL_TIME_ET="16:05")

def test_settings_step_scheduled_first(tmp_path):
    loop = dl.DeskLoop(_Engine(), config_fn=lambda: _cfg(tmp_path))
    names = [n for n, _, _ in loop._schedule(_cfg(tmp_path))]
    assert names[0] == "settings"

def test_settings_step_promotes_and_audits(tmp_path, monkeypatch):
    calls = []
    def fake_promote(manager, desk_dir, *, now):
        calls.append((desk_dir, now))
        return [{"key": "risk_per_trade_pct", "old": 0.01, "new": 0.02, "effective_at": "x"}]
    monkeypatch.setattr(dl, "promote_pending", fake_promote)
    eng = _Engine()
    loop = dl.DeskLoop(eng, config_fn=lambda: _cfg(tmp_path))
    now = datetime(2026, 9, 29, 9, 26, tzinfo=ET)
    asyncio.run(loop._step_settings(_cfg(tmp_path), now, "2026-09-29"))
    assert calls and calls[0][0] == str(tmp_path)
    assert any(a[0] == "settings_console" and a[1] == "promoted" for a in eng.audits)
```
(Use the loop class's real name from `core/loops/desk_loop.py` — check `class ` there; adjust `DeskLoop` if it differs. Other desk-loop tests under `tests/core/desk/` show the constructor pattern; mirror them.)

`tests/core/test_engine_block_entries.py` — the engine is too heavy to instantiate; assert the wiring textually and unit-test the audit reason:
```python
import inspect, re
def test_signal_router_checks_block_switch_before_any_other_gate():
    import core.engine as eng
    src = inspect.getsource(eng.TradingEngineWithCommentary._process_signal_with_commentary)
    received = src.index('"received"')
    block = src.index("BLOCK_NEW_ENTRIES")
    assert received < block
    assert "entries_blocked_by_console" in src
    # the block must come before the shadow meta-model call, i.e. before any other work
    assert block < src.index("_shadow_meta_evaluate")
```
(Confirm the engine class name via `grep -n "^class " core/engine.py`.)

- [ ] **Step 2: Run the three test files; expected FAIL** (attribute/assertion errors).
- [ ] **Step 3: Implement** the two properties, the engine block, and the desk-loop step.
- [ ] **Step 4: Run** the three files plus `tests/core/desk -q` and `tests/test_config.py -q`; all PASS.
- [ ] **Step 5: Commit**

```bash
git add core/config.py core/engine.py core/loops/desk_loop.py tests/test_config.py tests/core/desk/test_desk_loop_settings_step.py tests/core/test_engine_block_entries.py
git commit -m "feat(console): block-new-entries switch in the signal router; desk loop promotes pending settings at 09:25"
```

---

### Task 4: Console API router

**Files:**
- Create: `api/console.py`
- Modify: `api/routes.py` (register the router; find where `app` is created ~line 45–102 and add `from api.console import router as console_router` + `app.include_router(console_router)` after the middleware is added)
- Test: `tests/api/test_console_routes.py`

**Interfaces:**
- Consumes: Task 1 (`KNOBS`, `env_shadowed`, `yaml_value`, `effective_value`, `ALLOCATION_CEILING_PATH`, `DEFAULT_ALLOCATION_CEILING`), Task 2 (`submit_change`, `list_pending`, `cancel_pending`, `read_audit`, `set_block_new_entries`), Task 3 (`Config().BLOCK_NEW_ENTRIES`, `DESK_DATA_DIR`).
- Produces (all under `router = APIRouter(prefix="/api/console", tags=["console"])`; the app-level `APIAuthMiddleware` already enforces the bearer key, do not add a second dependency):
  - `GET /api/console` → `{"ok": true, "block_new_entries": bool, "allocation_ceiling": float, "promote_time_et": "09:25", "knobs": [ {key, label, group, kind, value, yaml_value, env_shadowed, restart_required, minimum, maximum, allowed, pending: {id,new,effective_at}|null} ... ], "pending": [...], "audit": [...last 20] }`; on any failure `{"ok": false, "error": str}` HTTP 200.
  - `PUT /api/console/{key}` body `{"value": Any, "reason": str}` → HTTP 200 `{"ok": true, "result": ChangeResult-as-dict}` for `applied|pending|noop`; HTTP 400 for `rejected` with a validation reason; HTTP 409 for `rejected` with reason starting `env_shadowed` or equal to `restart_required`; HTTP 404 for `unknown setting`.
  - `POST /api/console/block-entries` body `{"block": bool, "reason": str, "confirm": str?}` → 200 on applied, 400 on rejected.
  - `DELETE /api/console/pending/{pending_id}` → 200 `{"ok": true}` or 404.
  - `GET /api/console/audit?limit=50` → `{"ok": true, "rows": [...]}`.
  - `actor` is `"dashboard"` (the bearer key identifies the operator; do not log the key).
  - Every write route also calls `trading_engine._audit("settings_console", None, <event>, <key>, ...)` when `api.routes.trading_engine` is not None, wrapped in try/except (import `api.routes` lazily inside the function to avoid a circular import).
- Dependencies are resolved through two module-level factories that tests monkeypatch: `_manager()` returns `core.config.config_manager`, `_cfg()` returns `core.config.Config()`, `_now()` returns `core.loops.desk_steps.now_et()`.

- [ ] **Step 1: Tests** (`tests/api/test_console_routes.py`), following `_make_client` from `tests/api/test_desk_routes.py` (import `api.routes`, delenv `TRADING_API_KEY`, set `DESK_DATA_DIR` to `tmp_path`, `routes.trading_engine = None`). Monkeypatch `api.console._manager` to return a `FakeManager` (copy from Task 2 tests) and `api.console._cfg` to return a `SimpleNamespace` with `DESK_DATA_DIR=str(tmp_path)`, `BLOCK_NEW_ENTRIES=False`, `SETTINGS_PROMOTE_TIME_ET="09:25"`, `RISK_PER_TRADE_PCT=0.01`, `BOT_CAPITAL_ALLOCATION=5000.0`, `MAX_POSITIONS=5`, and the lane/gate properties; `api.console._now` to a fixed Monday 10:00 ET. Delete the env names of all knobs in an autouse fixture.

```python
def test_get_console_snapshot(client):
    body = client.get("/api/console").json()
    assert body["ok"] is True and body["block_new_entries"] is False
    keys = {k["key"] for k in body["knobs"]}
    assert "risk_per_trade_pct" in keys and "fit_lanes_enabled" in keys
    fit = next(k for k in body["knobs"] if k["key"] == "fit_lanes_enabled")
    assert fit["restart_required"] is True

def test_put_tighten_applies(client, manager):
    r = client.put("/api/console/risk_per_trade_pct", json={"value": 0.005, "reason": "cut"})
    assert r.status_code == 200 and r.json()["result"]["status"] == "applied"
    assert manager.data["trading.risk_per_trade_pct"] == 0.005

def test_put_loosen_pending_and_cancel(client):
    r = client.put("/api/console/risk_per_trade_pct", json={"value": 0.02, "reason": "more"})
    assert r.status_code == 200 and r.json()["result"]["status"] == "pending"
    pid = r.json()["result"]["pending_id"]
    snap = client.get("/api/console").json()
    knob = next(k for k in snap["knobs"] if k["key"] == "risk_per_trade_pct")
    assert knob["pending"]["id"] == pid
    assert client.delete(f"/api/console/pending/{pid}").status_code == 200
    assert client.delete(f"/api/console/pending/{pid}").status_code == 404

def test_put_validation_400(client):
    assert client.put("/api/console/risk_per_trade_pct", json={"value": 2, "reason": ""}).status_code == 400

def test_put_unknown_404(client):
    assert client.put("/api/console/nope", json={"value": 1, "reason": ""}).status_code == 404

def test_put_env_shadowed_409(client, monkeypatch):
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "0.01")
    r = client.put("/api/console/risk_per_trade_pct", json={"value": 0.005, "reason": ""})
    assert r.status_code == 409 and "env_shadowed" in r.json()["error"]
    snap = client.get("/api/console").json()
    assert next(k for k in snap["knobs"] if k["key"] == "risk_per_trade_pct")["env_shadowed"] is True

def test_restart_required_409(client):
    assert client.put("/api/console/fit_lanes_enabled", json={"value": "mr_20d", "reason": ""}).status_code == 409

def test_block_unblock(client, manager):
    assert client.post("/api/console/block-entries", json={"block": True, "reason": "x"}).status_code == 200
    assert manager.data["trading.block_new_entries"] is True
    assert client.post("/api/console/block-entries", json={"block": False, "reason": "x"}).status_code == 400
    assert client.post("/api/console/block-entries", json={"block": False, "reason": "x", "confirm": "YES"}).status_code == 200
    assert manager.data["trading.block_new_entries"] is False

def test_audit_endpoint(client):
    client.put("/api/console/risk_per_trade_pct", json={"value": 0.005, "reason": "cut"})
    rows = client.get("/api/console/audit?limit=5").json()["rows"]
    assert rows and rows[-1]["key"] == "risk_per_trade_pct"
```

- [ ] **Step 2: Run; expected 404s / import errors.**
- [ ] **Step 3: Implement `api/console.py`** (≤ 250 lines) and register the router in `api/routes.py`. Snapshot building lives in a helper `_knob_row(knob, manager, cfg, pending_by_key)`; keep each route < 30 lines.
- [ ] **Step 4: Run `tests/api -q` and `tests/test_api_auth.py -q`; all PASS.** Also confirm the app still imports: `/home/nvidia/anaconda3/envs/trading-bot/bin/python -c "import api.routes"`.
- [ ] **Step 5: Commit**

```bash
git add api/console.py api/routes.py tests/api/test_console_routes.py
git commit -m "feat(console): /api/console routes — snapshot, ratcheted writes, block switch, pending, audit"
```

---

### Task 5: Dashboard "Risk Console" card

**Files:**
- Modify: `templates/dashboard.html` — Settings tab markup after the hands-off card (~line 3700, before the "Info note" div) and JS next to the hands-off functions (~line 3794); the tab switch at ~line 3787 must also call `loadRiskConsole()` when `tabName === 'settings'`.
- Test: `tests/test_dashboard_console_markup.py` (new; string assertions on the template, mirroring any existing template test such as `tests/test_activity_page.py`).

**Behaviour:**
- Card `id="risk-console-card"`, title "Risk Console". Top row: a large toggle button `id="rc-block-btn"` that reads "BLOCK NEW ENTRIES" (red) when unblocked and "ENTRIES BLOCKED — click to resume" (grey) when blocked; resume prompts `window.prompt('Type YES to resume new entries')` and sends `confirm`. A second link-button "Stop bot" is NOT added (out of scope; `/api/stop` exists elsewhere).
- Three groups rendered from the `GET /api/console` snapshot in order risk, lanes, gates: each knob row has label, effective value, an input (number for money/fraction/int, checkbox for bool, text for csv), a reason input (placeholder "why"), an "Apply" button, and badges: `pending until <effective_at local>` with a small "cancel" link, `env override — file only` (disables the input), `restart required` (disables the input).
- Applying calls `PUT /api/console/{key}` via `authFetch`; shows the result line in `id="rc-status"` (green applied, amber pending with effective time, red rejected with the server message) and reloads the snapshot.
- Audit list `id="rc-audit"`: last 20 rows, newest first, one line each: `ts · event · key · old → new · reason`.
- No new external libraries; follow the existing card styles and `var(--…)` tokens used by the hands-off card.
- Keep all new JS inside one `// ============== RISK CONSOLE ==============` block, functions `loadRiskConsole`, `renderRiskConsole`, `applyRiskKnob`, `cancelRiskPending`, `toggleBlockEntries`, `renderRiskAudit`; escape all server strings through the template's existing HTML-escape helper if one exists (grep `escapeHtml`), else add a 4-line `rcEscape()`.

- [ ] **Step 1: Test** (`tests/test_dashboard_console_markup.py`):
```python
from pathlib import Path
def test_settings_tab_has_risk_console():
    html = Path("templates/dashboard.html").read_text()
    settings = html[html.index('id="tab-settings"'):]
    for needle in ('id="risk-console-card"', 'id="rc-block-btn"', 'id="rc-status"', 'id="rc-audit"',
                   'function loadRiskConsole', 'function applyRiskKnob', 'function toggleBlockEntries',
                   "/api/console"):
        assert needle in html, needle
    assert 'loadRiskConsole()' in html[html.index("tabName === 'settings'"):][:300]
```
- [ ] **Step 2: Run; FAIL.**
- [ ] **Step 3: Implement** markup + JS. Verify JS syntax by extracting the new script block to a temp file and running `node --check` if `node` is available (`which node`); otherwise load the page in the test client (`GET /`) and assert HTTP 200.
- [ ] **Step 4: Run the test + `tests/test_activity_page.py -q`; PASS.**
- [ ] **Step 5: Commit**

```bash
git add templates/dashboard.html tests/test_dashboard_console_markup.py
git commit -m "feat(console): Risk Console card on the Settings tab — block switch, ratcheted knobs, pending badges, audit"
```

---

### Task 6: Migration script, docs, changelog

**Files:**
- Create: `scripts/migrate_console_settings.py`
- Test: `tests/test_migrate_console_settings.py`
- Modify: `docs/BOT_OPERATIONS_MANUAL.md` (new §9.9 "Risk Console"), `docs/desk.md` (flags table: add `BLOCK_NEW_ENTRIES`, `SETTINGS_PROMOTE_TIME_ET`, the console files), `CHANGELOG.md` (entry `v-risk-console-2026-09-28`).

**Script behaviour** (`python scripts/migrate_console_settings.py --env .env --yaml "Config().yaml" [--apply]`):
- Dry-run by default: prints, for each console knob whose env name is present in the env file, `ENV_NAME=<value> → trading.<yaml_key>: <yaml literal>` and the `trading.settings_console.allocation_ceiling` line it will add (value = current `BOT_CAPITAL_ALLOCATION` env value, or 5000 when absent). Prints nothing else from the env file (no secrets).
- `--apply`: backs up both files to `<file>.bak_console_<YYYYmmdd_HHMMSS>`, writes the yaml keys via `ConfigManager(config_path=yaml).update(...)` (csv env values become lists), and comments out the migrated lines in the env file as `# migrated-to-yaml 2026-09-28: NAME=…` preserving all other lines byte-for-byte. `MAX_DAILY_LOSS` migrates only if its env value parses as a fraction in (0,1] (`.env` once held a dollar value); `DESK_STREAK_SINCE` and `FIT_LANES_ENABLED` are NOT migrated (file-only / restart-scoped) and are listed under "left in .env".
- Exit 0; exit 2 when the env file lacks every knob (nothing to do).

- [ ] **Step 1: Test** (`tests/test_migrate_console_settings.py`) with a temp `.env` containing `BOT_CAPITAL_ALLOCATION=5000`, `RISK_PER_TRADE_PCT=0.01`, `MAX_DAILY_LOSS=0.03`, `FIT_LANE_LIVE_ALLOWLIST="mr_20d:long,mom_12_1:long:risk_on|mixed"`, `FIT_LANES_ENABLED=mr_20d,mom_12_1`, `DESK_STREAK_SINCE=2026-09-28`, `ORB_LIVE_ENTRIES_ENABLED=1`, `TRADING_API_KEY=secret-xyz` and a temp yaml with `trading: {}`:
  - dry-run output mentions `BOT_CAPITAL_ALLOCATION` and never contains `secret-xyz`;
  - after `--apply`: yaml has `trading.bot_capital_allocation == 5000.0`, `trading.fit_lane_live_allowlist == ["mr_20d:long", "mom_12_1:long:risk_on|mixed"]`, `trading.orb_live_entries_enabled is True`, `trading.settings_console.allocation_ceiling == 5000.0`; env file still contains `FIT_LANES_ENABLED=mr_20d,mom_12_1`, `DESK_STREAK_SINCE=2026-09-28` and `TRADING_API_KEY=secret-xyz` unchanged, and `BOT_CAPITAL_ALLOCATION` only in a `# migrated-to-yaml` comment; backups exist.
  - Import the script's `migrate(env_path, yaml_path, apply: bool, now: datetime) -> dict` function directly rather than spawning a process.
- [ ] **Step 2: Run; FAIL.**
- [ ] **Step 3: Implement the script (≤ 200 lines), then the docs.** Docs content for §9.9: what the console is, the ratchet rule with the 09:25 promotion, what is file-only and why (streak reset, allocation ceiling, `FIT_LANES_ENABLED`), the two runtime files under `data/desk/`, the env-shadowing rule and the migration command, the block switch audit line `engine_decision component=signal_router … reason=entries_blocked_by_console`.
- [ ] **Step 4: Run the test; PASS.** `git diff --check`.
- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_console_settings.py tests/test_migrate_console_settings.py docs/BOT_OPERATIONS_MANUAL.md docs/desk.md CHANGELOG.md
git commit -m "feat(console): env→yaml migration script; operations manual and desk docs for the Risk Console"
```

---

### After the plan (controller, after 16:00 ET, not a subagent task)

1. Run the full targeted suites once more. 2. `python scripts/migrate_console_settings.py --env .env --yaml "Config().yaml"` dry-run, review, then `--apply`. 3. Restart the bot via the established plain-python launch, `POST /api/start`, `POST /api/toggle-mode live`. 4. `GET /api/console` and confirm no knob reports `env_shadowed`, the values match the go-live numbers, and `block_new_entries` is false. 5. Record in `.superpowers/sdd/2026-09-28-risk-console/progress.md` and memory.
