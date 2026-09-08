"""Catalyst-momentum v2 walk-forward (T9 / G1).

v-catalyst-momentum-v2-wf-2026-09-03. Context (operator, 2026-09-03):
bot took one flat SNAP mean-rev trade while day-one movers ran without
it. Log autopsy: breakout evaluated ~4,700 setups, fired ZERO —
(a) strict close>high_20 trigger missed GLXY/RKT/BULL/HSAI pressing
within 1% of highs on 4-6.5x volume; (b) ADX>=25 floor structurally
excludes day-one breakouts (BIAF +32% above high_20 skipped at ADX
16.7; BMNR missed by 0.6 ADX) because ADX lags fresh trends.

Prior art: momentum_ladder_walkforward_2026-06-12 — momentum entries +
ladder exits ran PF 1.03-1.27 in trending windows but 0.63-0.84 in
choppy ones and stayed dead. Its own decision gate prescribed the
untested fix: allocator gating (trend tapes only).

v2 entry (long only, NO ADX floor):
    close >= high_20 * 0.99          # day-one proximity trigger
    volume_ratio >= 2.0              # real participation
    (high - low) >= 1.3 * atr        # range expansion
    close > sma_20                   # uptrend context
    rsi <= 80                        # parabolic guard
    [pass B only] er_20 >= 0.30      # Kaufman efficiency-ratio regime
                                     # gate — chop produces no trades,
                                     # not losses

Exits: the live ladder (BE at +0.5R, 1.5xATR trail after +2xATR),
bar-conservative, copied from the June script. Stop 1.5xATR, target
3xATR (2R).

Passes over the standard six 60-day windows:
    A. v2 entries, ladder exits, no regime gate
    B. v2 entries, ladder exits, er_20 >= 0.30 gate

SHIP CRITERION (decided before running): pass B must show aggregate
PF >= 1.5 AND no individual window PF < 1.0 (windows with zero trades
are fine — that is the regime gate doing its job). Otherwise the
strategy stays dead and this report documents why.

Run:
    /home/nvidia/anaconda3/envs/trading-bot/bin/python \
        research/catalyst_momentum_v2_walkforward_2026-09-03.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.getLogger("TradingBot").setLevel(logging.WARNING)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "walkforward_base", REPO_ROOT / "research" / "walkforward_2026-06-09.py"
)
_wf = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_wf)

import numpy as np
import pandas as pd

import backtest.engine as bt_engine
from backtest.engine import (DEFAULT_DSN, MAX_HOLD_BARS,
                             _run_backtest_with_trades, compute_indicators,
                             _run_strategy)
from backtest.trades import Trade
from core.models import MarketData, SignalType, TradingSignal
from strategies.builtin import MomentumStrategyWithCommentary

FREQUENCY = 5
# Ladder parameters — mirror live defaults (same as June script).
BE_ACTIVATION_R = 0.5
TRAIL_ACT_ATR = 2.0
TRAIL_WIDTH_ATR = 1.5

# v2 entry parameters (documented in module docstring).
PROX = 0.99
VOL_MIN = 2.0
RANGE_ATR = 1.3
RSI_MAX = 80.0
ER_PERIOD = 20
ER_MIN = 0.30
STOP_ATR = 1.5
TARGET_ATR = 3.0

OUT_PATH = REPO_ROOT / "research" / "catalyst_momentum_v2_report_2026-09-03.json"

# Toggled between passes; read by the patched entry generator.
_ER_GATE_ON = False

_orig_generate = MomentumStrategyWithCommentary.generate_signal_with_commentary


async def _v2_generate(self, market_data) -> TradingSignal | None:
    """v2 catalyst-momentum entry. Replaces the MACD/ADX momentum
    entry for the duration of this walk-forward. No commentary calls —
    research context."""
    ind = market_data.indicators
    try:
        close = float(market_data.close)
        atr = float(ind.get("atr") or 0)
        high_20 = float(ind.get("high_20") or 0)
        sma_20 = float(ind.get("sma_20") or 0)
        vol_ratio = float(ind.get("volume_ratio") or 0)
        rsi = float(ind.get("rsi") or 50)
        bar_range = float(market_data.high) - float(market_data.low)
    except (TypeError, ValueError):
        return None

    if atr <= 0 or high_20 <= 0 or sma_20 <= 0:
        return None
    if close < high_20 * PROX:
        return None
    if vol_ratio < VOL_MIN:
        return None
    if bar_range < RANGE_ATR * atr:
        return None
    if close <= sma_20:
        return None
    if rsi > RSI_MAX:
        return None
    if _ER_GATE_ON:
        er = ind.get("er_20")
        try:
            er = float(er)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(er) or er < ER_MIN:
            return None

    return TradingSignal(
        symbol=market_data.symbol,
        signal_type=SignalType.BUY,
        strength=0.7,
        entry_price=close,
        stop_loss=close - STOP_ATR * atr,
        take_profit=close + TARGET_ATR * atr,
        position_size=0,
        reasoning={"strategy": "momentum", "variant": "catalyst_v2",
                   "atr": atr, "er_20": ind.get("er_20"),
                   "volume_ratio": vol_ratio},
        confidence=0.6,
    )


def _add_er(df: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    """Kaufman efficiency ratio over ER_PERIOD bars, computed on
    completed bars up to and including the current one (same
    convention as every other indicator here — no lookahead)."""
    close = df["Close"]
    diff_abs = close.diff().abs()
    denom = diff_abs.rolling(ER_PERIOD).sum()
    num = (close - close.shift(ER_PERIOD)).abs()
    er = num / denom.replace(0.0, np.nan)
    out = ind.copy()
    out["er_20"] = er
    return out


async def replay_symbol_ladder_er(symbol, df, strategies):
    """June's ladder replay + er_20 injection. Bar-conservative: stop
    checks use the PRIOR bar's ratcheted stop; ladder updates after
    exit checks. Long-only model."""
    ind = _add_er(df, compute_indicators(df))
    trades: list[Trade] = []
    open_pos = {name: None for name, _ in strategies}
    _low_arr = df["Low"].to_numpy()

    for i in range(50, len(df)):
        bar = df.iloc[i]
        indicators = ind.iloc[i].to_dict()
        if any(pd.isna(v) for k, v in indicators.items()
               if k in ("rsi", "macd", "macd_signal", "bb_lower", "bb_middle",
                        "bb_upper", "atr", "adx", "sma_20", "sma_50",
                        "volume_ratio", "high_20", "low_20")):
            continue
        indicators["lows_50"] = _low_arr[max(0, i - 50):i].tolist()

        md = MarketData(
            symbol=symbol, timestamp=df.index[i].to_pydatetime(),
            open=float(bar["Open"]), high=float(bar["High"]),
            low=float(bar["Low"]), close=float(bar["Close"]),
            volume=float(bar["Volume"]), indicators=indicators,
            timeframe="5m",
        )

        for name, strategy in strategies:
            pos = open_pos[name]
            if pos is not None:
                pos["hold"] += 1
                stop_hit = bar["Low"] <= pos["stop"]
                target_hit = bar["High"] >= pos["target"]
                exit_reason = exit_price = None
                if stop_hit:
                    exit_price = pos["stop"]
                    exit_reason = ("trail" if pos["stop"] > pos["orig_stop"]
                                   else "stop")
                elif target_hit:
                    exit_price = pos["target"]
                    exit_reason = "target"
                elif pos["hold"] >= MAX_HOLD_BARS:
                    exit_price = float(bar["Close"])
                    exit_reason = "timeout"
                if exit_reason is not None:
                    trades.append(Trade(
                        strategy=name, symbol=symbol, side=pos["side"],
                        entry_time=pos["entry_time"],
                        entry_price=pos["entry"],
                        exit_time=df.index[i], exit_price=float(exit_price),
                        exit_reason=exit_reason, hold_bars=pos["hold"],
                        entry_indicators=pos.get("entry_indicators", {}),
                    ))
                    open_pos[name] = None
                    pos = None
                else:
                    if pos["side"] == "long":
                        peak = max(pos["peak"], float(bar["High"]))
                        pos["peak"] = peak
                        stop_dist = pos["entry"] - pos["orig_stop"]
                        if (stop_dist > 0
                                and peak >= pos["entry"]
                                + BE_ACTIVATION_R * stop_dist):
                            pos["stop"] = max(pos["stop"], pos["entry"])
                        atr = pos["atr"]
                        if atr > 0 and peak >= pos["entry"] + TRAIL_ACT_ATR * atr:
                            new_trail = float(bar["Close"]) - TRAIL_WIDTH_ATR * atr
                            pos["stop"] = max(pos["stop"], new_trail)

            if pos is None:
                try:
                    signal = await _run_strategy(strategy, md)
                except Exception:
                    signal = None
                if signal is None:
                    continue
                entry_ind = {}
                for k, v in indicators.items():
                    if isinstance(v, (list, tuple)):
                        continue
                    try:
                        if not pd.isna(v):
                            entry_ind[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                open_pos[name] = {
                    "side": "long" if signal.signal_type == SignalType.BUY
                            else "short",
                    "entry": float(signal.entry_price),
                    "stop": float(signal.stop_loss),
                    "orig_stop": float(signal.stop_loss),
                    "target": float(signal.take_profit),
                    "entry_time": df.index[i],
                    "hold": 0,
                    "peak": float(signal.entry_price),
                    "atr": float(indicators.get("atr", 0) or 0),
                    "entry_indicators": entry_ind,
                }
    return trades


def _window_list():
    windows = []
    end = _wf.DATA_END
    for _ in range(_wf.N_WINDOWS):
        start = end - timedelta(days=_wf.WINDOW_DAYS)
        windows.append((start, end))
        end = start
    windows.reverse()
    return windows


async def run_pass(label: str, dsn: str) -> list[dict]:
    results = []
    for i, (start, w_end) in enumerate(_window_list(), 1):
        symbols = _wf.window_symbols(dsn, start, w_end, _wf.TOP_N)
        report, _trades = await _run_backtest_with_trades(
            symbols=symbols, days=_wf.WINDOW_DAYS, frequency=FREQUENCY,
            dsn=dsn, bars_loader=_wf.make_window_loader(dsn, w_end),
            run_id=f"cmv2_{label}{i}", seed=42,
        )
        mom = report.get("per_strategy", {}).get("momentum", {})
        m = mom.get("all", mom) or {}
        results.append({"window": i, **{k: m.get(k) for k in (
            "n_trades", "win_rate_pct", "profit_factor",
            "sum_return_pct", "exit_breakdown", "avg_hold_bars")}})
        n = m.get("n_trades") or 0
        if n:
            print(f"  [{label}] w{i}: n={n:4} PF={m['profit_factor']:5.2f} "
                  f"WR={m['win_rate_pct']:5.1f}% "
                  f"sum={m['sum_return_pct']:+7.1f}%", flush=True)
        else:
            print(f"  [{label}] w{i}: n=0", flush=True)
    return results


async def main() -> None:
    global _ER_GATE_ON
    dsn = DEFAULT_DSN

    MomentumStrategyWithCommentary.generate_signal_with_commentary = _v2_generate
    _orig_replay = bt_engine.replay_symbol
    bt_engine.replay_symbol = replay_symbol_ladder_er
    try:
        print("pass A: v2 entries + ladder exits, no regime gate", flush=True)
        _ER_GATE_ON = False
        a = await run_pass("nogate", dsn)

        print(f"pass B: v2 entries + ladder exits + er_20 >= {ER_MIN}",
              flush=True)
        _ER_GATE_ON = True
        b = await run_pass("ergate", dsn)

        out = {
            "generated": datetime.now().isoformat(),
            "entry_params": {"prox": PROX, "vol_min": VOL_MIN,
                             "range_atr": RANGE_ATR, "rsi_max": RSI_MAX,
                             "er_period": ER_PERIOD, "er_min": ER_MIN,
                             "stop_atr": STOP_ATR, "target_atr": TARGET_ATR},
            "ladder_params": {"be_r": BE_ACTIVATION_R,
                              "trail_act_atr": TRAIL_ACT_ATR,
                              "trail_width_atr": TRAIL_WIDTH_ATR},
            "ship_criterion": "pass B aggregate PF >= 1.5 AND no window PF < 1.0",
            "no_gate": a, "er_gate": b,
        }
        OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
        print(f"report → {OUT_PATH}", flush=True)
    finally:
        MomentumStrategyWithCommentary.generate_signal_with_commentary = _orig_generate
        bt_engine.replay_symbol = _orig_replay


if __name__ == "__main__":
    asyncio.run(main())
