"""Intraday Lab — Standalone FastAPI server on port 8005.

Run: python -m intraday_lab.app
Does NOT import or modify gap_fade_app.py.
"""

import asyncio
import json
import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure parent on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from .adapter import get_all_strategies, list_strategy_ids, get_strategy_params
from .backtester import IntradayBacktester
from .walkforward import IntradayWalkForward
from .combiner import StrategyCombiner

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('IntradayLab')

DB_URL = os.environ.get('DATABASE_URL', 'postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev')
PORT = int(os.environ.get('INTRADAY_LAB_PORT', 8005))

app = FastAPI(title='Intraday Lab', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# Running task state
_task_status = {'status': 'idle', 'message': '', 'progress': 0, 'result': None}


def _progress(pct, msg):
    _task_status['progress'] = round(pct, 1)
    _task_status['message'] = msg


# ── Health ──
@app.get('/api/lab/health')
async def health():
    return {'status': 'ok', 'port': PORT}


# ── List Strategies ──
@app.get('/api/lab/strategies')
async def strategies():
    ids = list_strategy_ids()
    result = []
    for sid in ids:
        params = get_strategy_params(sid)
        result.append({'id': sid, 'params': params})
    return {'strategies': result}


# ── Strategy Params ──
@app.get('/api/lab/strategies/{strategy_id}/params')
async def strategy_params(strategy_id: str):
    params = get_strategy_params(strategy_id)
    return {'strategy_id': strategy_id, 'params': params}


# ── Run Single Backtest ──
@app.post('/api/lab/backtest')
async def run_backtest(body: dict):
    global _task_status
    if _task_status['status'] == 'running':
        return {'error': 'A task is already running'}

    strategy_id = body.get('strategy_id', '')
    if not strategy_id:
        return {'error': 'strategy_id required'}

    _task_status = {'status': 'running', 'message': 'Starting...', 'progress': 0, 'result': None}

    async def _run():
        try:
            bt = IntradayBacktester(
                DB_URL,
                capital=body.get('capital', 100000),
                risk_pct=body.get('risk_pct', 0.01),
                max_positions=body.get('max_positions', 3),
            )
            result = bt.run(
                strategy_id,
                body.get('start_date', '2025-01-01'),
                body.get('end_date', '2026-03-29'),
                symbols=body.get('symbols'),
                config=body.get('config'),
                start_time=body.get('start_time', '10:00'),
                end_time=body.get('end_time', '15:50'),
                max_symbols_per_day=body.get('max_symbols_per_day', 20),
                progress_callback=_progress,
            )
            _task_status['status'] = 'complete'
            _task_status['result'] = result
            _task_status['progress'] = 100
        except Exception as e:
            _task_status['status'] = 'error'
            _task_status['message'] = str(e)
            logger.error(f'Backtest error: {e}', exc_info=True)

    asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(_run())))
    return {'status': 'started', 'strategy_id': strategy_id}


# ── Parameter Sweep ──
@app.post('/api/lab/sweep')
async def run_sweep(body: dict):
    global _task_status
    if _task_status['status'] == 'running':
        return {'error': 'A task is already running'}

    strategy_id = body.get('strategy_id', '')
    param_grid = body.get('param_grid', {})
    if not strategy_id or not param_grid:
        return {'error': 'strategy_id and param_grid required'}

    _task_status = {'status': 'running', 'message': 'Starting sweep...', 'progress': 0, 'result': None}

    def _run_sync():
        try:
            bt = IntradayBacktester(
                DB_URL,
                capital=body.get('capital', 100000),
                risk_pct=body.get('risk_pct', 0.01),
                max_positions=body.get('max_positions', 3),
            )
            results = bt.sweep(
                strategy_id,
                body.get('start_date', '2025-01-01'),
                body.get('end_date', '2026-03-29'),
                param_grid,
                symbols=body.get('symbols'),
                max_symbols_per_day=body.get('max_symbols_per_day', 20),
                progress_callback=_progress,
            )
            _task_status['status'] = 'complete'
            _task_status['result'] = {'results': results, 'count': len(results)}
            _task_status['progress'] = 100
        except Exception as e:
            _task_status['status'] = 'error'
            _task_status['message'] = str(e)

    asyncio.get_event_loop().run_in_executor(None, _run_sync)
    return {'status': 'started'}


# ── Walk-Forward ──
@app.post('/api/lab/walkforward')
async def run_walkforward(body: dict):
    global _task_status
    if _task_status['status'] == 'running':
        return {'error': 'A task is already running'}

    strategy_id = body.get('strategy_id', '')
    param_grid = body.get('param_grid', {})
    if not strategy_id or not param_grid:
        return {'error': 'strategy_id and param_grid required'}

    _task_status = {'status': 'running', 'message': 'Starting walk-forward...', 'progress': 0, 'result': None}

    def _run_sync():
        try:
            wf = IntradayWalkForward(DB_URL, capital=body.get('capital', 100000))
            result = wf.run(
                strategy_id, param_grid,
                body.get('start_date', '2025-01-01'),
                body.get('end_date', '2026-03-29'),
                in_sample_days=body.get('in_sample_days', 60),
                out_sample_days=body.get('out_sample_days', 15),
                step_days=body.get('step_days', 15),
                symbols=body.get('symbols'),
                max_symbols_per_day=body.get('max_symbols_per_day', 20),
                progress_callback=_progress,
            )
            _task_status['status'] = 'complete'
            _task_status['result'] = result
            _task_status['progress'] = 100
        except Exception as e:
            _task_status['status'] = 'error'
            _task_status['message'] = str(e)

    asyncio.get_event_loop().run_in_executor(None, _run_sync)
    return {'status': 'started'}


# ── Combine Strategies ──
@app.post('/api/lab/combine')
async def run_combine(body: dict):
    global _task_status
    if _task_status['status'] == 'running':
        return {'error': 'A task is already running'}

    strategies = body.get('strategies', [])
    if not strategies:
        return {'error': 'strategies list required'}

    _task_status = {'status': 'running', 'message': 'Starting combination test...', 'progress': 0, 'result': None}

    def _run_sync():
        try:
            c = StrategyCombiner(
                DB_URL,
                capital=body.get('capital', 100000),
                risk_pct=body.get('risk_pct', 0.01),
                max_positions=body.get('max_positions', 5),
            )
            result = c.run(
                strategies,
                body.get('start_date', '2025-01-01'),
                body.get('end_date', '2026-03-29'),
                symbols=body.get('symbols'),
                max_symbols_per_day=body.get('max_symbols_per_day', 20),
            )
            _task_status['status'] = 'complete'
            _task_status['result'] = result
            _task_status['progress'] = 100
        except Exception as e:
            _task_status['status'] = 'error'
            _task_status['message'] = str(e)

    asyncio.get_event_loop().run_in_executor(None, _run_sync)
    return {'status': 'started'}


# ── Task Status ──
@app.get('/api/lab/status')
async def task_status():
    return _task_status


# ── Results ──
@app.get('/api/lab/results')
async def task_results():
    return _task_status.get('result') or {'error': 'No results'}


# ── Cancel ──
@app.post('/api/lab/cancel')
async def cancel():
    global _task_status
    _task_status = {'status': 'idle', 'message': 'Cancelled', 'progress': 0, 'result': None}
    return {'status': 'cancelled'}


# ── Run All Strategies (ranking) ──
@app.post('/api/lab/rank-all')
async def rank_all(body: dict):
    """Run all intraday strategies and rank by performance."""
    global _task_status
    if _task_status['status'] == 'running':
        return {'error': 'A task is already running'}

    _task_status = {'status': 'running', 'message': 'Ranking all strategies...', 'progress': 0, 'result': None}

    def _run_sync():
        try:
            ids = list_strategy_ids()
            results = []
            for i, sid in enumerate(ids):
                _progress((i + 1) / len(ids) * 100, f'Testing {sid} ({i+1}/{len(ids)})')
                bt = IntradayBacktester(DB_URL, capital=body.get('capital', 100000))
                result = bt.run(
                    sid,
                    body.get('start_date', '2025-01-01'),
                    body.get('end_date', '2026-03-29'),
                    max_symbols_per_day=body.get('max_symbols_per_day', 20),
                )
                if result and 'error' not in result:
                    summary = {k: v for k, v in result.items() if k != 'trades'}
                    results.append(summary)

            results.sort(key=lambda r: r.get('sharpe', 0), reverse=True)

            # Assign status
            for r in results:
                sh = r.get('sharpe', 0)
                wr = r.get('win_rate', 0)
                pf = r.get('profit_factor', 0)
                if sh >= 2.0 and pf >= 1.5:
                    r['recommendation'] = 'ready'
                elif sh >= 1.0 and pf >= 1.0:
                    r['recommendation'] = 'marginal'
                else:
                    r['recommendation'] = 'reject'

            _task_status['status'] = 'complete'
            _task_status['result'] = {'rankings': results, 'count': len(results)}
            _task_status['progress'] = 100
        except Exception as e:
            _task_status['status'] = 'error'
            _task_status['message'] = str(e)

    asyncio.get_event_loop().run_in_executor(None, _run_sync)
    return {'status': 'started'}


# ── Visual Replay via WebSocket ──
from fastapi import WebSocket, WebSocketDisconnect
from .engine import ReplayEngine, BarLoader, IndicatorState, LabBar

@app.websocket('/api/lab/replay')
async def replay_ws(ws: WebSocket):
    """Stream bar-by-bar replay with trade signals over WebSocket.

    Client sends: {"action": "start", "strategy_id": "orb_breakout",
                   "date": "2026-03-10", "symbols": ["TSLA","NVDA"],
                   "speed": 4, "config": {...}}
    Server streams: {"type": "bar", "symbol": "TSLA", "ts": ..., "o":, "h":, "l":, "c":, "v":}
                    {"type": "trade", "symbol": "TSLA", "action": "buy", "price": 180.50, ...}
                    {"type": "indicator", "symbol": "TSLA", "vwap": 181.2, "rsi": 42, ...}
                    {"type": "done", "trades": [...], "metrics": {...}}
    """
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            action = msg.get('action')

            if action == 'start':
                strategy_id = msg.get('strategy_id', 'orb_breakout')
                date = msg.get('date', '2026-03-10')
                end_date = msg.get('end_date', '')  # empty = single day
                symbols = msg.get('symbols', ['TSLA', 'NVDA', 'AAPL', 'META'])
                speed = msg.get('speed', 4)  # bars per second
                config = msg.get('config')

                strategy = None
                try:
                    from .adapter import create_strategy
                    strategy = create_strategy(strategy_id, config)
                except Exception as e:
                    await ws.send_json({'type': 'error', 'message': f'Strategy load failed: {e}'})
                    continue

                if not strategy:
                    await ws.send_json({'type': 'error', 'message': f'Strategy {strategy_id} not found'})
                    continue

                loader = BarLoader(DB_URL)

                # Multi-day: get list of trading dates
                if end_date and end_date > date:
                    trading_dates = loader.get_trading_dates(date, end_date)
                else:
                    trading_dates = [date]

                # Aggregate all bars across all days
                all_bars: dict = {}
                for td in trading_dates:
                    day_bars = loader.load_day_batch(symbols, td, '09:30', '15:50')
                    for sym, bars in day_bars.items():
                        all_bars.setdefault(sym, []).extend(bars)

                if not all_bars:
                    await ws.send_json({'type': 'error', 'message': f'No minute bars for {date}. Check if data is backfilled.'})
                    loader.close()
                    continue

                # Warn about missing symbols
                missing = [sym for sym in symbols if sym not in all_bars]
                if missing:
                    await ws.send_json({'type': 'warning', 'message': f'No minute data for: {", ".join(missing)}'})

                # Build unified timeline
                timeline = []
                for sym, bars in all_bars.items():
                    for bar in bars:
                        timeline.append(bar)
                timeline.sort(key=lambda b: b.ts)

                # Send symbol list + metadata
                await ws.send_json({
                    'type': 'init',
                    'symbols': list(all_bars.keys()),
                    'date': date,
                    'end_date': end_date or date,
                    'days': len(trading_dates),
                    'strategy': strategy_id,
                    'total_bars': len(timeline),
                    'speed': speed,
                })

                current_day = ''

                # Replay engine state
                indicators = {sym: IndicatorState() for sym in all_bars}
                positions = {}  # sym -> position dict
                trades = []
                equity = 100000
                risk_pct = 0.01

                delay = 1.0 / speed if speed > 0 else 0

                for i, bar in enumerate(timeline):
                    sym = bar.symbol
                    indicators[sym].update(bar)

                    # Day change notification
                    bar_day = bar.ts.strftime('%Y-%m-%d')
                    if bar_day != current_day:
                        current_day = bar_day
                        # Reset indicators for new day
                        for s in indicators:
                            indicators[s].reset()
                        # Close any positions from previous day (EOD)
                        for s in list(positions.keys()):
                            last_price = bar.close if s == sym else 0
                            if last_price > 0:
                                pos = positions[s]
                                if pos['direction'] == 'long':
                                    pnl = (last_price - pos['entry']) * pos['shares']
                                else:
                                    pnl = (pos['entry'] - last_price) * pos['shares']
                                equity += pnl
                                trades.append({'symbol': s, 'direction': pos['direction'], 'entry': pos['entry'], 'exit': last_price, 'pnl': round(pnl, 2), 'reason': 'eod', 'entry_time': pos['entry_time'], 'exit_time': bar.ts.isoformat(), 'shares': pos['shares']})
                                await ws.send_json({'type': 'trade', 'action': 'exit', 'symbol': s, 'price': round(last_price, 4), 'direction': pos['direction'], 'pnl': round(pnl, 2), 'reason': 'eod', 'ts': bar.ts.isoformat(), 'equity': round(equity, 2)})
                                del positions[s]
                        await ws.send_json({'type': 'day_change', 'date': bar_day, 'equity': round(equity, 2), 'trades_so_far': len(trades)})

                    # Send bar
                    await ws.send_json({
                        'type': 'bar',
                        'symbol': sym,
                        'ts': bar.ts.isoformat(),
                        'o': round(bar.open, 4),
                        'h': round(bar.high, 4),
                        'l': round(bar.low, 4),
                        'c': round(bar.close, 4),
                        'v': bar.volume,
                        'i': i,
                    })

                    # Send indicators
                    tick_data = indicators[sym].get_tick_data()
                    await ws.send_json({
                        'type': 'indicator',
                        'symbol': sym,
                        'vwap': round(tick_data.get('vwap', 0), 4),
                        'rsi': round(tick_data.get('rsi', 50), 1),
                        'ema9': round(tick_data.get('ema9', 0), 4),
                        'ema21': round(tick_data.get('ema21', 0), 4),
                        'atr': round(tick_data.get('atr', 0), 4),
                        'vol_surge': round(tick_data.get('volume_surge', 1), 2),
                    })

                    # Check exits
                    if sym in positions:
                        pos = positions[sym]
                        exited = False
                        exit_price = 0
                        exit_reason = ''

                        if pos['direction'] == 'long' and bar.low <= pos['stop']:
                            exit_price, exit_reason, exited = pos['stop'], 'stop_loss', True
                        elif pos['direction'] == 'short' and bar.high >= pos['stop']:
                            exit_price, exit_reason, exited = pos['stop'], 'stop_loss', True
                        elif pos['direction'] == 'long' and bar.high >= pos['target']:
                            exit_price, exit_reason, exited = pos['target'], 'target', True
                        elif pos['direction'] == 'short' and bar.low <= pos['target']:
                            exit_price, exit_reason, exited = pos['target'], 'target', True

                        if exited:
                            if pos['direction'] == 'long':
                                pnl = (exit_price - pos['entry']) * pos['shares']
                            else:
                                pnl = (pos['entry'] - exit_price) * pos['shares']
                            equity += pnl

                            trade = {
                                'symbol': sym, 'direction': pos['direction'],
                                'entry': pos['entry'], 'exit': exit_price,
                                'pnl': round(pnl, 2), 'reason': exit_reason,
                                'entry_time': pos['entry_time'], 'exit_time': bar.ts.isoformat(),
                                'shares': pos['shares'],
                            }
                            trades.append(trade)
                            del positions[sym]

                            await ws.send_json({
                                'type': 'trade',
                                'action': 'exit',
                                'symbol': sym,
                                'price': round(exit_price, 4),
                                'direction': pos['direction'],
                                'pnl': round(pnl, 2),
                                'reason': exit_reason,
                                'ts': bar.ts.isoformat(),
                                'equity': round(equity, 2),
                            })

                    # Check entries (after 30 bars for indicators to warm up)
                    max_pos = msg.get('max_positions', 10)
                    if sym not in positions and len(indicators[sym].bars) >= 30 and len(positions) < max_pos:
                        try:
                            setup = strategy.scan_for_setups(
                                sym, tick_data,
                                {'price': bar.close, 'volume': bar.volume},
                                bar.ts)
                            if setup:
                                valid, _ = strategy.validate_setup(setup, tick_data, bar.ts)
                                if valid:
                                    risk_amount = equity * risk_pct
                                    risk_per_share = abs(setup.entry_price - setup.stop_price)
                                    shares = max(1, int(risk_amount / risk_per_share)) if risk_per_share > 0 else 1
                                    shares = min(shares, int(equity * 0.2 / setup.entry_price)) if setup.entry_price > 0 else 1

                                    positions[sym] = {
                                        'direction': setup.direction,
                                        'entry': setup.entry_price,
                                        'stop': setup.stop_price,
                                        'target': setup.target_price,
                                        'shares': shares,
                                        'entry_time': bar.ts.isoformat(),
                                    }

                                    await ws.send_json({
                                        'type': 'trade',
                                        'action': 'entry',
                                        'symbol': sym,
                                        'price': round(setup.entry_price, 4),
                                        'direction': setup.direction,
                                        'stop': round(setup.stop_price, 4),
                                        'target': round(setup.target_price, 4),
                                        'shares': shares,
                                        'confidence': round(setup.confidence, 2),
                                        'ts': bar.ts.isoformat(),
                                        'equity': round(equity, 2),
                                    })
                        except Exception:
                            pass

                    # Throttle for visual effect
                    if delay > 0 and i % max(1, speed) == 0:
                        await asyncio.sleep(delay)

                # Force close remaining positions at last bar price
                for sym, pos in list(positions.items()):
                    last_bars = all_bars.get(sym, [])
                    if last_bars:
                        exit_price = last_bars[-1].close
                        if pos['direction'] == 'long':
                            pnl = (exit_price - pos['entry']) * pos['shares']
                        else:
                            pnl = (pos['entry'] - exit_price) * pos['shares']
                        equity += pnl
                        trades.append({
                            'symbol': sym, 'direction': pos['direction'],
                            'entry': pos['entry'], 'exit': exit_price,
                            'pnl': round(pnl, 2), 'reason': 'eod',
                            'entry_time': pos['entry_time'],
                            'exit_time': last_bars[-1].ts.isoformat(),
                            'shares': pos['shares'],
                        })
                        await ws.send_json({
                            'type': 'trade', 'action': 'exit', 'symbol': sym,
                            'price': round(exit_price, 4), 'direction': pos['direction'],
                            'pnl': round(pnl, 2), 'reason': 'eod',
                            'ts': last_bars[-1].ts.isoformat(), 'equity': round(equity, 2),
                        })

                # Done
                wins = [t for t in trades if t['pnl'] > 0]
                losses = [t for t in trades if t['pnl'] <= 0]
                gp = sum(t['pnl'] for t in wins)
                gl = abs(sum(t['pnl'] for t in losses))

                await ws.send_json({
                    'type': 'done',
                    'trades': trades,
                    'metrics': {
                        'total_trades': len(trades),
                        'wins': len(wins),
                        'losses': len(losses),
                        'win_rate': len(wins) / len(trades) if trades else 0,
                        'net_pnl': round(sum(t['pnl'] for t in trades), 2),
                        'profit_factor': round(gp / gl, 2) if gl > 0 else 999,
                        'final_equity': round(equity, 2),
                    },
                })

                loader.close()

            elif action == 'stop':
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({'type': 'error', 'message': str(e)})
        except:
            pass


if __name__ == '__main__':
    import uvicorn
    logger.info(f'Starting Intraday Lab on port {PORT}')
    uvicorn.run(app, host='0.0.0.0', port=PORT)
