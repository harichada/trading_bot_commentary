# Backtest Screen — Design

**Date:** 2026-05-04
**Branch:** feature/trading_bot_v1
**Author:** Claude (brainstorming, user-approved)

## Problem

The Backtesting tab in `templates/dashboard.html` exists but every "Run
Backtest" click 500s — `api/routes.py:1621` does
`from backtesting_engine import BacktestingEngine, BacktestConfig, BacktestMode`
and that module is not in the tree. Meanwhile the script that actually
produces every `backtest_report*.json` artifact on disk —
`backtest_strategies.py` — is not reachable from the UI at all. The screen
needs to drive the existing per-strategy replay against Postgres
`minute_bars` and render the per-strategy quality report.

## Decisions (locked during brainstorming)

| | |
|---|---|
| Engine | Per-strategy replay via `backtest_strategies.py` (the same script that produced existing `backtest_report*.json`) |
| Run mode | On-demand only (no saved-report viewer in scope) |
| Symbol input | Radio toggle: Top-N or explicit comma-separated list |
| Results detail | Per-strategy summary + exit-reason stacked bars + per-symbol drill-down |
| Execution | Refactor script into importable `run_backtest(...)` + emit progress over WebSocket |
| Storage | Every run writes `backtest_results/run_YYYYMMDD_HHMMSS.json` and updates `backtest_results/latest_report.json` |
| Concurrency | Single-flight: 2nd concurrent request returns 429 |

## Architecture

Three units, communicate by data:

```
templates/dashboard.html  ──POST /api/backtest/run─►  api/backtest.py
   (Backtesting tab JS)   ◄───── run_id ─────────       │ (single-flight,
        │                                               │  WS progress,
        └──── WS /ws/backtest/{run_id} ─────────────────┤  saves JSON)
                                                        │
                                                        ▼
                                          backtest_strategies.py
                                          (pure async; no FastAPI;
                                           accepts progress_cb +
                                           injectable bars_loader)
```

Why split `api/backtest.py` out of `routes.py`: `routes.py` is already 3,298
lines and the project's coding rule is ≤800. Backtest deserves its own
module; cleanup of dead code in `routes.py` (~300 lines removed) lands at
the same time.

## Contracts

### `POST /api/backtest/run`

Request:
```json
{
  "symbol_mode": "top_n" | "explicit",
  "top_n": 30,                     // when symbol_mode="top_n"
  "symbols": ["NVDA","TSLA"],      // when symbol_mode="explicit"
  "days": 90,
  "frequency": 5                   // 1 | 5 | 15 minutes
}
```

Validation at boundary: `top_n ∈ [1,200]`, `days ∈ [1,365]`,
`frequency ∈ {1,5,15}`, symbol regex `^[A-Z.]{1,8}$`.

Response (returns immediately, ~50ms):
```json
{ "status": "started", "run_id": "bt_20260504_082146_a1b2" }
```

If a backtest is already in progress: `429` with
`{ "error": "backtest_running", "run_id": "<existing>" }`.

### `WS /ws/backtest/{run_id}`

Server → client events (envelope: `{type, run_id, ...}`):

| `type` | Payload | When |
|---|---|---|
| `progress` | `{stage, symbols_done, symbols_total, current_symbol}` | After each symbol; stage ∈ {`loading_data`, `replaying`} |
| `warning` | `{message}` | Non-fatal (no bars for X, etc.) |
| `complete` | `{report_path, report}` | Run finished; full BacktestReport dict |
| `error` | `{message}` | Fatal; run aborted |

Client closes the WS after `complete` or `error`.

### `GET /api/backtest/latest`

Returns most recent `backtest_results/latest_report.json` so a refreshed
tab doesn't lose the last result. 404 if no run has ever completed.

### `BacktestReport` dataclass (matches existing JSON shape, additive `per_symbol`)

```python
@dataclass(frozen=True)
class SymbolMetrics:
    n_trades: int
    win_rate_pct: float
    sum_return_pct: float
    avg_return_pct: float

@dataclass(frozen=True)
class StrategyMetrics:
    # existing fields preserved verbatim from current backtest_report.json
    n_trades: int
    win_rate_pct: float
    avg_return_pct: float
    median_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    sum_return_pct: float
    max_drawdown_pct: float
    per_trade_sharpe: float
    profit_factor: float | None
    avg_hold_bars: float
    exit_breakdown: dict[str, float]      # {stop, target, timeout, eod}
    per_symbol: dict[str, SymbolMetrics]  # NEW — for drill-down

@dataclass(frozen=True)
class BacktestReport:
    run_id: str
    timestamp: str                        # ISO8601
    symbols: list[str]
    symbol_count: int
    days: int
    frequency_minutes: int
    total_trades: int
    elapsed_seconds: float
    per_strategy: dict[str, dict[str, StrategyMetrics]]  # name → {all, long, short}
```

`per_symbol` is purely additive; all old reports remain readable.

## Files touched

| File | Change | Approx LoC |
|---|---|---|
| `backtest_strategies.py` | Refactor: extract `async def run_backtest(...)`, add `per_symbol`, frozen dataclasses, keep CLI as wrapper | ~+80 / ~-30 |
| `api/backtest.py` | NEW — endpoint + WS + single-flight tracker | ~+220 |
| `api/routes.py` | Import `api.backtest`; delete `simulate_backtest` (165), broken `/api/backtest/run` (170), `/api/professional/backtest*` (~270), `/api/backtest/status` (5) | ~-610 / +5 |
| `templates/dashboard.html` | Replace tab inputs, replace results panes, replace `runBacktest()` + `drawEquityCurve()` with new flow that uses Chart.js + per-symbol drill-down | ~+200 / -150 |
| `tests/test_backtest_strategies.py` | NEW — synthetic bars_loader fixture, asserts shape + progress_cb invoked | ~+120 |

## UI

The tab keeps its shell; inputs and result panes change.

```
┌─ 🔬 Strategy Backtesting ──────────────────────────────────────┐
│  [Configuration]                                                │
│  Symbol selection: ( ) Top-N most active   ( ) Explicit list   │
│   Top-N: [30]   Days: [90]   Frequency: [5min ▾]               │
│   (Symbols box appears when "Explicit list" picked)             │
│   [🚀 Run Backtest]                                             │
│   Progress: [████████░░] 80% — replaying NVDA (24/30)           │
├─────────────────────────────────────────────────────────────────┤
│  [Per-Strategy Summary]                                         │
│   strategy        trades  win%  avg%   sumR%  pf    sharpe  DD% │
│   breakout        2,202   38.0  -0.27  -599   0.76  -0.09  -652 │
│   mean_reversion  1,412   51.9  +0.04  +50    1.04  +0.01  -104 │
│   momentum        3,165   37.6  -0.28  -875   0.78  -0.08  -954 │
│   (click row → expand per-symbol table below)                   │
├─────────────────────────────────────────────────────────────────┤
│  [Exit Reasons]                                                 │
│   Stacked bars per strategy: stop / target / timeout / eod      │
└─────────────────────────────────────────────────────────────────┘
```

## Error handling

| Failure | Surface |
|---|---|
| Validation error | 400 with `{"error":"<field>:<reason>"}` |
| Backtest already running | 429 with existing `run_id` |
| Postgres unreachable | WS `error` event, run-tracker cleared, button re-enabled |
| Single symbol load fails | WS `warning` event, continue with others |
| WS dropped mid-run | Run continues server-side; reconnecting client gets nothing more for that run, but `latest_report.json` is updated on completion and `GET /api/backtest/latest` shows the result |

## Tests

Unit (pytest):
- `run_backtest()` with synthetic bars_loader returns `BacktestReport` with the
  expected per_strategy keys (`breakout`, `mean_reversion`, `momentum`) and
  per_symbol drill-down populated.
- `progress_cb` invoked once per symbol with monotonically increasing
  `symbols_done`.
- Frozen dataclass: assignment to a field raises.

API:
- `POST /api/backtest/run` returns `run_id` immediately.
- Second concurrent POST returns 429.
- Validation: `frequency=7` → 400.
- `GET /api/backtest/latest` returns 404 when no report exists.

Skipped (manual smoke): full Postgres path + WebSocket browser flow.
